"""The Dedalus core on the shared IR contracts
(docs/ir_boundary_and_constraints.md): tensor unknowns, LapH, the
divergence cleaning of a written involution, and the composite boundary
rows on the shell, each against a closed form.

  * tensor unknowns: the IR has no radial derivative of a rank-2 unknown
    (Grad and Lap refuse it), so a tensor lifts no tau; T - strain(v) = 0
    keeps T in the state and reproduces the strain-form spectrum, the
    relaxed dt(T) + T - strain(v) = 0 keeps T's DOFs finite and turns
    every toroidal decay rate Ek k^2 into the pair of roots of
    lambda^2 - lambda + Ek k^2 = 0, and rows written on T are rows on the
    derivative of its vector, lifted on that vector;
  * LapH = lap - (1/r^2) d_r(r^2 d_r): the operator identity on
    r^(l+2) Y_lm on shell and ball, the pointwise identity at the ball's
    nodes on the singular image LapH(z) = -2 z / r^2, and the complement
    lap - LapH as pure radial diffusion, whose Dirichlet shell spectrum is
    (n pi / (r_o - r_i))^2 once per harmonic;
  * divergence cleaning: free decay in a conducting shell and ball with
    Div(b) = 0 written and no multiplier gets the record's auto multiplier
    (the pencil of the hand-written one, entry for entry), the Bessel decay
    rates to spectral accuracy, div b and the multiplier at rounding on
    the eigenvectors; an implied involution and a hand-written multiplier
    compile as written;
  * composite rows on the shell, resolved term by term as on the ball:
    Robin and Biot walls (a 2 x 2 Bessel determinant), Navier slip on the
    toroidal family, the traction-free Stokes shell (a 4 x 4 determinant
    over the poloidal basis), every coefficient spelling, and the refusals
    of a theta-dependent coefficient, a mixed tangential row and a lone
    tangential projection.

Skips when Dedalus is not importable.
"""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

pytest.importorskip("dedalus")

import sympy as sp

from analytic.manufactured import _dedalus_state
from spdee import Background, Problem, compile_problem, r, theta
from spdee.applications.solar_eigenmodes import add_magnetic_field
from spdee.backends.dedalus import DedalusGrid
from spdee.solvers import dense, shift_invert

RI = 0.5
BALL = (DedalusGrid(N_r=10, N_theta=12), DedalusGrid(N_r=14, N_theta=16))
SHELL = (DedalusGrid(N_r=12, N_theta=12, r_inner=RI), DedalusGrid(N_r=18, N_theta=16, r_inner=RI))
GRIDS = {"ball": BALL, "shell": SHELL}
STRAIN = "dot(grad(v) + trans(grad(v)), r_hat)"


def _roots(f, n=2, kmin=1e-2, kmax=24.0):
    """The first n positive roots of the scalar function f."""
    ks = np.linspace(kmin, kmax, 12000)
    v = np.array([f(k) for k in ks])
    out = []
    for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        out.append(brentq(f, ks[i], ks[i + 1]))
    return np.array(out[:n])


def _det2(rows):
    return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]


def _jl(ell, k, rw, d=False):
    return spherical_jn(ell, k * rw, derivative=d)


def _yl(ell, k, rw, d=False):
    return spherical_yn(ell, k * rw, derivative=d)


def _census(cp, sigma):
    """Finite pencil eigenvalues, asserted real, and the infinite count."""
    K, M = cp.assemble_evp_matrices()
    res = dense(K, M, sigma=sigma, backend="numpy")
    assert np.abs(res.lam.imag).max() < 1e-8 * max(1.0, np.abs(res.lam).max())
    return np.sort(res.lam.real), res.n_infinite


def _nearest_errors(lam, targets):
    return np.array([np.abs(lam - t).min() / abs(t) for t in targets])


def _walls(geometry):
    return ("outer",) if geometry == "ball" else ("outer", "inner")


def _spectral(errs, tol):
    """Two-resolution errors: below tol at the finer one and not growing."""
    assert errs[1].max() < tol, errs
    assert errs[1].max() <= errs[0].max() + 1e-12, errs


# ---- tensor unknowns

def _viscous(viscous, geometry, dt_T=False, rows_on="v"):
    """dt(v) + 2 ez x v + grad(p) - Ek viscous = 0, div(v) = 0 with
    stress-free walls; T is defined by T - strain(v) = 0, under Dt with
    ``dt_T``; ``rows_on='T'`` writes the tangential stress rows on T."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    if "T" in viscous:
        pb.add_unknown("T", "tensor")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_parameter("Ek", 1e-2, symbolic=False)
    pb.add_equation(f"dt(v) + 2*cross(ez, v) + grad(p) - Ek*({viscous}) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    if "T" in viscous:
        pb.add_equation(("dt(T) + " if dt_T else "") + "T - grad(v) - trans(grad(v)) = 0",
                        name="def")
    for at in _walls(geometry):
        if rows_on == "v":
            pb.add_stress_free("v", at=at)
        else:
            pb.add_bc("dot(r_hat, v) = 0", at=at)
            pb.add_bc("dot(theta_hat, dot(T, r_hat)) = 0", at=at)
            pb.add_bc("dot(phi_hat, dot(T, r_hat)) = 0", at=at)
    return pb


def _near(cp, sigma, radius=0.8):
    K, M = cp.assemble_evp_matrices()
    res = dense(K, M, sigma=sigma, backend="numpy")
    lam = res.lam
    return np.sort_complex(lam[np.abs(lam - sigma) < radius]), res.n_infinite


@pytest.mark.parametrize("geometry", ["ball", "shell"])
@pytest.mark.parametrize("rows_on", ["v", "T"])
def test_definition_keeps_the_tensor(geometry, rows_on):
    """T - strain(v) = 0 keeps T in the state with no tau of its own (the
    tau layout is the strain form's) and its finite spectrum is the
    strain form's: to rounding on the ball, and on the shell to the
    difference of the tau polynomials of the two formulations, which
    decreases spectrally with resolution.  Rows written on T are hosted
    on v and give the same spectrum."""
    sigma = 0.3j
    errs, sizes = [], []
    for grid in GRIDS[geometry]:
        cpS = _viscous("div(grad(v) + trans(grad(v)))", geometry).compile(
            grid, m=1, backend="dedalus", gauge_unknowns=["p"])
        cpT = _viscous("div(T)", geometry, rows_on=rows_on).compile(
            grid, m=1, backend="dedalus", gauge_unknowns=["p"])
        assert [t.name for t in cpT.tau_fields] == [t.name for t in cpS.tau_fields]
        a, nS = _near(cpS, sigma)
        b, nT = _near(cpT, sigma)
        assert a.size == b.size >= 3
        n_T = cpT.A_total.shape[0] - cpS.A_total.shape[0]
        assert nT == nS + n_T                                     # T's rows carry no mass
        errs.append(np.abs(a - b).max())
        sizes.append(n_T)
    assert sizes[1] > sizes[0]
    if geometry == "ball":
        assert max(errs) < 1e-12, errs
    else:
        assert errs[1] < 1e-9 and errs[1] < errs[0], errs


def _toroidal_roots(ell, n=2):
    """Zero tangential stress on the toroidal profile j_l(k r) at r = 1:
    k j_l'(k) = j_l(k)."""
    return _roots(lambda k: k * _jl(ell, k, 1.0, True) - _jl(ell, k, 1.0), n=n)


@pytest.mark.parametrize("m", [0, 1])
def test_relaxed_tensor_splits_every_decay_rate(m):
    """dt(v) - Ek div(T) = 0 with dt(T) + T - strain(v) = 0 on the
    stress-free ball: T keeps its DOFs (the infinite count is the strain
    form's), and on a toroidal Bessel mode div(strain(v)) = -k^2 v, so
    T = strain(v) / (1 + s) turns each rate Ek k^2 of the diffusion
    problem into the two roots of lambda^2 - lambda + Ek k^2 = 0 (the
    rigid rotation k = 0 gives 0 and 1); complex beyond
    4 Ek k^2 = 1."""
    Ek = 0.01
    for grid in BALL:
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_unknown("T", "tensor")
        pb.add_parameter("Ek", Ek)
        pb.add_equation("dt(v) - Ek*div(T) = 0", name="momentum")
        pb.add_equation("dt(T) + T - grad(v) - trans(grad(v)) = 0", name="relax")
        pb.add_stress_free("v", at="outer")
        cp = pb.compile(grid, m=m, backend="dedalus")
        assert [t.name for t in cp.tau_fields] == ["tau_v1"]
        ref = Problem()
        ref.add_unknown("v", "vector")
        ref.add_parameter("Ek", Ek)
        ref.add_equation("dt(v) - Ek*div(grad(v) + trans(grad(v))) = 0", name="momentum")
        ref.add_stress_free("v", at="outer")
        ref = ref.compile(grid, m=m, backend="dedalus")
        K, M = cp.assemble_evp_matrices()
        res = dense(K, M, sigma=0.5, backend="numpy")
        assert res.n_infinite == dense(*ref.assemble_evp_matrices(), sigma=0.5,
                                       backend="numpy").n_infinite
        lam = res.lam
        ks = [0.0] if m <= 1 else []
        for ell in range(max(m, 1), 4):
            ks.extend(_toroidal_roots(ell))
        for k in ks:
            for root in np.roots([1.0, -1.0, Ek * k ** 2]):
                assert np.abs(lam - root).min() < 1e-9, (k, root)


def test_tensor_rows_need_a_vector():
    """A tensor whose equation differentiates no vector cannot host wall
    rows: it has no radial derivative to lift them on."""
    pb = Problem()
    pb.add_unknown("T", "tensor")
    pb.add_equation("dt(T) + T = 0")
    pb.add_bc("dot(r_hat, dot(T, r_hat)) = 0", at="outer")
    with pytest.raises(NotImplementedError, match="no radial derivative"):
        pb.compile(BALL[0], m=1, backend="dedalus")


# ---- LapH

def _laph_problem():
    pb = Problem()
    pb.add_unknown("u", "scalar")
    pb.add_equation("dt(u) - laph(u) = 0")
    pb.add_bc("u = 0", at="outer")
    return pb


def _wall_factor(geometry):
    return 1 - r ** 2 if geometry == "ball" else (1 - r) * (r - RI)


def _identity_residual(cp, u, image):
    """|K x_u + M x_image| / |M x_image| for K = -LapH, M the mass; u
    vanishes at the walls so the boundary rows of K are satisfied."""
    K, M = cp.assemble_evp_matrices()
    x_u = _dedalus_state(cp, {"u": u})
    x_im = _dedalus_state(cp, {"u": image})
    rhs = M @ x_im
    return np.abs(K @ x_u + rhs).max() / np.abs(rhs).max()


@pytest.mark.parametrize("geometry", ["ball", "shell"])
@pytest.mark.parametrize("m", [0, 1, 2])
def test_laph_identity(geometry, m):
    """LapH(f Y_lm) = -l(l+1) f / r^2 Y_lm on f = r^(l+2) w(r), w the wall
    factor, l = m + 1, regular on the ball, to spectral accuracy on both
    geometries: the pencil of dt(u) - LapH(u) = 0 applied to the field
    reproduces the image through the mass matrix.  On the shell the
    default ncc_cutoff of 1e-8 truncates the 1/r^2 NCC at that level."""
    ell = m + 1
    ylm = sp.sin(theta) ** m * sp.cos(theta)
    u = r ** (ell + 2) * _wall_factor(geometry) * ylm
    image = -ell * (ell + 1) * r ** ell * _wall_factor(geometry) * ylm
    cp = _laph_problem().compile(GRIDS[geometry][1], m=m, backend="dedalus")
    assert _identity_residual(cp, u, image) < 1e-7            # the shell's 1/r^2 NCC cutoff
    cp = _laph_problem().compile(GRIDS[geometry][1], m=m, backend="dedalus", ncc_cutoff=1e-14)
    assert _identity_residual(cp, u, image) < 1e-11


def test_laph_on_the_ball_is_pointwise_at_the_nodes():
    """LapH(z (1 - r^2)) = -2 z (1 - r^2) / r^2 is singular at the origin:
    the ball LapH is the pointwise operator at the quadrature nodes (its
    1/r^2 factor is probed in grid space), so the identity holds there to
    rounding although the image is outside the basis."""
    cp = _laph_problem().compile(BALL[1], m=0, backend="dedalus")
    assert len(cp._aux) == 1
    z = r * sp.cos(theta)
    assert _identity_residual(cp, z * (1 - r ** 2), -2 * z * (1 - r ** 2) / r ** 2) < 1e-12


@pytest.mark.parametrize("m", [0, 1, 2])
def test_lap_minus_laph_is_radial_diffusion_on_the_shell(m):
    """dt(u) - (lap(u) - LapH(u)) = 0 is u_rr + 2 u_r / r with Dirichlet
    walls: u = sin(k (r - r_i)) / r, k = n pi / (r_o - r_i), so
    lambda = k^2 for every harmonic l >= m alike, each with multiplicity
    N_theta - m, converging spectrally with N_r."""
    errs = []
    for grid in SHELL:
        pb = Problem()
        pb.add_unknown("u", "scalar")
        pb.add_equation("dt(u) - lap(u) + laph(u) = 0")
        pb.add_bc("u = 0", at="inner")
        pb.add_bc("u = 0", at="outer")
        cp = pb.compile(grid, m=m, backend="dedalus")
        lam, n_inf = _census(cp, 10.0)
        assert n_inf == 4 * (grid.N_theta - m)
        worst = 0.0
        for n in (1, 2, 3):
            k2 = (n * np.pi / (1.0 - RI)) ** 2
            d = np.sort(np.abs(lam - k2) / k2)
            assert d[grid.N_theta - m - 1] < 1e-5 < d[grid.N_theta - m], (n, d[:20])
            worst = max(worst, d[grid.N_theta - m - 1])
        errs.append(np.array([worst]))
    _spectral(errs, 1e-9)


# ---- divergence cleaning

def _decay(geometry, multiplier=None):
    """Free decay dt(b) = lap(b) in a perfect conductor, Div(b) = 0
    written; with ``multiplier`` the hand-written saddle point."""
    pb = Problem()
    pb.add_unknown("b", "vector")
    if multiplier:
        pb.add_unknown(multiplier, "scalar")
        pb.add_equation(f"dt(b) - lap(b) + grad({multiplier}) = 0", name="induction")
    else:
        pb.add_equation("dt(b) - lap(b) = 0", name="induction")
    pb.add_equation("div(b) = 0", name="solenoidal")
    for at in _walls(geometry):
        pb.add_bc("dot(r_hat, b) = 0", at=at)
        pb.add_bc("dot(theta_hat, curl(b)) = 0", at=at)
        pb.add_bc("dot(phi_hat, curl(b)) = 0", at=at)
    return pb


def _decay_targets(geometry, m, n=2):
    """lambda = k^2 over l = max(m, 1), max(m, 1) + 1: the toroidal family
    closes on d(r T)/dr = 0 at every wall, the poloidal on S = 0."""
    out = []
    for ell in range(max(m, 1), max(m, 1) + 2):
        if geometry == "ball":
            tor = lambda k, ell=ell: _jl(ell, k, 1.0) + k * _jl(ell, k, 1.0, True)
            pol = lambda k, ell=ell: _jl(ell, k, 1.0)
        else:
            def tor(k, ell=ell):
                return _det2([[_jl(ell, k, rw) + k * rw * _jl(ell, k, rw, True),
                               _yl(ell, k, rw) + k * rw * _yl(ell, k, rw, True)]
                              for rw in (RI, 1.0)])

            def pol(k, ell=ell):
                return _det2([[_jl(ell, k, rw), _yl(ell, k, rw)] for rw in (RI, 1.0)])
        out.extend(k ** 2 for k in _roots(tor, n=n))
        out.extend(k ** 2 for k in _roots(pol, n=n))
    return np.sort(out)


@pytest.mark.parametrize("geometry", ["ball", "shell"])
@pytest.mark.parametrize("m", [0, 1])
def test_written_involution_is_cleaned(geometry, m):
    """Div(b) = 0 beside the curl evolution and no multiplier: the core
    takes the record's DivergenceCleaning, an auto multiplier hosted as a
    gauge scalar (+ Grad(__div_b) in the induction equation, integ at
    l = 0), and the pencil is the hand-written multiplier's entry for
    entry with the identity permutation (the multiplier is appended after
    b as psi is); the decay rates are the Bessel ones to spectral
    accuracy, div b and the multiplier vanish on the eigenvectors."""
    targets = _decay_targets(geometry, m)
    errs = []
    for grid in GRIDS[geometry]:
        cp = _decay(geometry).compile(grid, m=m, backend="dedalus")
        ref = _decay(geometry, "psi").compile(grid, m=m, backend="dedalus", gauge_unknowns=["psi"])
        assert cp.multipliers == {"b": "__div_b"}
        assert cp.unknown_names == ["b", "__div_b"] and ref.unknown_names == ["b", "psi"]
        assert [t.name for t in cp.tau_fields] == [
            t.name.replace("psi", "__div_b") for t in ref.tau_fields]
        assert abs(cp.A_total - ref.A_total).max() == 0.0
        assert abs(cp.B_total - ref.B_total).max() == 0.0
        lam, _n = _census(cp, 0.5 * targets[0])
        errs.append(_nearest_errors(lam, targets))
        lam2, V = shift_invert(cp.A_total, cp.B_total, targets[0] + 1e-3, 3, return_vectors=True)
        cp.set_state(V[:, int(np.argmin(np.abs(lam2 - targets[0])))])
        b = cp.fields["b"]
        div = cp.d3.div(b).evaluate()
        div.change_scales(1)
        scale = np.abs(b["g"]).max()
        assert np.abs(div["g"]).max() < 1e-12 * scale
        assert np.abs(cp.fields["__div_b"]["g"]).max() < 1e-11 * scale
    _spectral(errs, 1e-8 if geometry == "shell" else 1e-12)


def test_free_decay_shell_targets_of_the_analytic_suite():
    """The 'b' formulation of tests/test_analytic_evp.py's free-decay
    benchmark (eta = 1, the same walls) at its Dedalus resolutions:
    toroidal l = 1 (k = 1.984570, 6.618465) and poloidal l = 1, 2
    (k = 6.572013, 7.111576) within its 1e-6, so the
    ``("b", "dedalus"): "Non-square system"`` refusal there no longer
    holds and that cell should run the convergence check."""
    targets = np.array([1.984570, 6.572013, 7.111576, 6.618465]) ** 2
    errs = []
    for grid in (DedalusGrid(N_r=14, N_theta=12, r_inner=RI),
                 DedalusGrid(N_r=20, N_theta=16, r_inner=RI)):
        lam, _n = _census(_decay("shell").compile(grid, m=0, backend="dedalus"), 30.0)
        errs.append(_nearest_errors(lam, targets))
    _spectral(errs, 1e-6)


def test_implied_involution_and_hand_written_multiplier_compile_as_written():
    """No written constraint: no multiplier is introduced (Div Curl = 0
    holds on the polynomial space and the parity matrix keeps the implied
    case unconstrained).  A hand-written multiplier beside a curl
    evolution is outside the classifier's vocabulary (both readings) and
    compiles as before, with its own gauge tau."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("b", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
    pb.add_equation("dt(b) - curl(cross(v, B0)) = 0", name="induction")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    cp = pb.compile(BALL[0], m=1, backend="dedalus", gauge_unknowns=["p"])
    assert cp.multipliers == {} and cp.unknown_names == ["v", "p", "b"]
    inertial = Problem()
    inertial.add_unknown("v", "vector")
    inertial.add_unknown("p", "scalar")
    inertial.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    inertial.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0", name="momentum")
    inertial.add_equation("div(v) = 0", name="continuity")
    inertial.add_bc("dot(r_hat, v) = 0", at="outer")
    B0 = Background.Vector("B0", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero)
    system = add_magnetic_field(inertial.build(gauge_unknowns=["p"]), B0, Le=0.3, eta=1e-2,
                                geometry="ball", solenoidal=True)
    cp = compile_problem(system, backend="dedalus", grid=BALL[0], m=1)
    assert cp.multipliers == {}
    assert [t.name for t in cp.tau_fields] == ["tau_v1", "tau_b1", "tau_p_gauge", "tau_psi_gauge"]


# ---- composite rows on the shell

def _helmholtz(bc, params=None, backgrounds=None, coefficient=None):
    pb = Problem()
    pb.add_unknown("p", "scalar")
    for name, value in (params or {}).items():
        pb.add_parameter(name, value)
    for name, expr in (backgrounds or {}).items():
        pb.add_background(name, expr=expr)
    if coefficient is None:
        pb.add_equation("dt(p) - lap(p) = 0")
    else:
        pb.add_parameter("kappa", coefficient)
        pb.add_equation("dt(p) - kappa*lap(p) = 0")
    for at in ("inner", "outer"):
        pb.add_bc(bc, at=at)
    return pb


def _robin_targets(m, h, n=2):
    """p = (a j_l(k r) + b y_l(k r)) Y_lm with k j_l'(k r_w) + h j_l(k r_w)
    = 0 (and y_l alike) at both walls: lambda = k^2 over l >= m.  The
    same r_hat in both rows makes the inner wall anti-dissipative, so a
    growing mode exists at m = 0 (k imaginary, not listed here)."""
    out = []
    for ell in range(m, m + 3):
        def det(k, ell=ell):
            return _det2([[k * _jl(ell, k, rw, True) + h * _jl(ell, k, rw),
                           k * _yl(ell, k, rw, True) + h * _yl(ell, k, rw)] for rw in (RI, 1.0)])
        out.extend(k ** 2 for k in _roots(det, n=n))
    return np.sort(out)


@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize("h", [2, 5])
def test_helmholtz_robin_shell(m, h):
    """dot(r_hat, grad(p)) + h*p = 0 at both walls (h = 3 would put the
    exact null mode a + b/r, h = 1/r_i + 1/r_o, at lambda = 0)."""
    targets = _robin_targets(m, h)
    errs = []
    for grid in SHELL:
        cp = _helmholtz("dot(r_hat, grad(p)) + h*p = 0", params={"h": float(h)}).compile(
            grid, m=m, backend="dedalus")
        assert [t.name for t in cp.tau_fields] == ["tau_p1", "tau_p2"]
        lam, n_inf = _census(cp, 1.0)
        assert n_inf == 4 * (grid.N_theta - m)
        errs.append(_nearest_errors(lam, targets))
    _spectral(errs, 1e-9)


def test_thermal_diffusion_biot_shell():
    """dt(S) - kappa*lap(S) = 0 with dot(r_hat, grad(S)) + Bi*S = 0 at both
    walls: lambda = kappa k^2 with the Robin determinant."""
    kappa, Bi = 0.1, 2.0
    targets = kappa * _robin_targets(1, Bi)
    errs = []
    for grid in SHELL:
        cp = _helmholtz("dot(r_hat, grad(p)) + Bi*p = 0", params={"Bi": Bi},
                        coefficient=kappa).compile(grid, m=1, backend="dedalus")
        lam, _n = _census(cp, 1.0)
        errs.append(_nearest_errors(lam, targets))
    _spectral(errs, 1e-9)


@pytest.mark.parametrize("bc,background", [
    ("dot(r_hat, grad(p)) + h*p = 0", None),
    ("dot(r_hat, grad(p)) + (2*r**2 - 3*r + 3)*p = 0", None),
    ("f0*p + dot(r_hat, grad(p)) = 0", "2*r**2 - 3*r + 3"),
    ("-(dot(r_hat, grad(p)) + 2*p) = 0", None),
    ("dot(grad(p), r_hat) - (-2)*p = 0", None),
], ids=["symbol", "sympy_r", "background", "negated", "reversed"])
def test_robin_coefficient_at_the_wall_shell(bc, background):
    """Every spelling of the coefficient is taken at the wall on the shell
    as on the ball: a symbol, a SymPy expression in r (2 r^2 - 3 r + 3 is
    2 at both walls), the same as a radial background interpolated there,
    a negated row, the unit vector on the right; all give the h = 2
    pencil."""
    ref = _helmholtz("dot(r_hat, grad(p)) + 2*p = 0").compile(SHELL[0], m=1, backend="dedalus")
    backgrounds = None if background is None else {"f0": background}
    cp = _helmholtz(bc, params={"h": 2.0}, backgrounds=backgrounds).compile(
        SHELL[0], m=1, backend="dedalus")
    scale = abs(ref.A_total).max()
    A, B, RA, RB = cp.A_total, cp.B_total, ref.A_total, ref.B_total
    if bc.startswith("-"):
        A, B, RA, RB = abs(A), abs(B), abs(RA), abs(RB)
    assert abs(A - RA).max() < 1e-13 * scale
    assert abs(B - RB).max() < 1e-13 * scale


def _slip_targets(m, ls, Ek, n=2):
    """Toroidal T = a j_l(k r) + b y_l(k r) with
    T + l_s (T' - T/r) = 0 at both walls: rates Ek k^2."""
    out = []
    for ell in range(max(m, 1), max(m, 1) + 3):
        def det(k, ell=ell):
            def row(f, f1, rw):
                return f + ls * (k * f1 - f / rw)
            return _det2([[row(_jl(ell, k, rw), _jl(ell, k, rw, True), rw),
                           row(_yl(ell, k, rw), _yl(ell, k, rw, True), rw)] for rw in (RI, 1.0)])
        out.extend(Ek * k ** 2 for k in _roots(det, n=n))
    return np.sort(out)


@pytest.mark.parametrize("m", [0, 1])
def test_vector_diffusion_navier_slip_shell(m):
    """The theta/phi pair of mixed rows dot(t_hat, v) + ls*dot(t_hat,
    dot(strain, r_hat)) = 0 at both walls is one angular row per wall,
    resolved term by term (the tangential unit vector is not an NCC of the
    sphere), with the tau layout of the stress-free shell."""
    Ek, ls = 0.01, 0.5
    targets = _slip_targets(m, ls, Ek)
    errs = []
    for grid in SHELL:
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_parameter("Ek", Ek)
        pb.add_parameter("ls", ls)
        pb.add_equation("dt(v) - Ek*lap(v) = 0")
        for at in ("inner", "outer"):
            pb.add_bc("dot(r_hat, v) = 0", at=at)
            for hat in ("theta_hat", "phi_hat"):
                pb.add_bc(f"dot({hat}, v) + ls*dot({hat}, {STRAIN}) = 0", at=at)
        cp = pb.compile(grid, m=m, backend="dedalus")
        assert [t.name for t in cp.tau_fields] == ["tau_v1", "tau_v2"]
        lam, _n = _census(cp, 0.1)
        errs.append(_nearest_errors(lam, targets))
    _spectral(errs, 1e-9)


def _traction_free_roots(ell, c, n=2):
    """Poloidal modes of the Stokes shell with traction-free walls,
    lap(v) + k^2 v = grad(p), div(v) = 0, lambda = k^2, on the basis
    grad(r^l Y), grad(r^(-l-1) Y), curl(curl(r j_l(k r) Y)) and the y_l
    twin: v_r = R Y, v_theta = S dY/dtheta with (R, S) =
    (l r^(l-1), r^(l-1)), (-(l+1) r^(-l-2), r^(-l-2)) and
    (l(l+1) f/r, f' + f/r), p = k^2 r^l Y, k^2 r^(-l-1) Y, 0, 0.  The rows
    2 R' - c p = 0 and S' - S/r + R/r = 0 at both walls have a solution
    when the 4 x 4 determinant vanishes."""
    def det(k):
        A = []
        for rw in (RI, 1.0):
            cols = [(ell * rw ** (ell - 1), ell * (ell - 1) * rw ** (ell - 2),
                     rw ** (ell - 1), (ell - 1) * rw ** (ell - 2), k ** 2 * rw ** ell),
                    (-(ell + 1) * rw ** (-ell - 2), (ell + 1) * (ell + 2) * rw ** (-ell - 3),
                     rw ** (-ell - 2), -(ell + 2) * rw ** (-ell - 3), k ** 2 * rw ** (-ell - 1))]
            for fn in (_jl, _yl):
                f, f1 = fn(ell, k, rw), k * fn(ell, k, rw, True)
                f2 = -(2 / rw) * f1 - (k ** 2 - ell * (ell + 1) / rw ** 2) * f
                R, R1 = ell * (ell + 1) * f / rw, ell * (ell + 1) * (f1 / rw - f / rw ** 2)
                S, S1 = f1 + f / rw, f2 + f1 / rw - f / rw ** 2
                cols.append((R, R1, S, S1, 0.0))
            A.append([2 * R1 - c * p for R, R1, S, S1, p in cols])
            A.append([S1 - S / rw + R / rw for R, R1, S, S1, p in cols])
        return np.linalg.det(np.array(A))
    return _roots(det, n=n, kmin=0.5)


def _toroidal_shell_roots(ell, n=2):
    """d/dr (T/r) = 0 at both walls on T = a j_l(k r) + b y_l(k r)."""
    return _roots(lambda k: _det2([[k * _jl(ell, k, rw, True) - _jl(ell, k, rw) / rw,
                                    k * _yl(ell, k, rw, True) - _yl(ell, k, rw) / rw]
                                   for rw in (RI, 1.0)]), n=n)


@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize("c", [1, 2])
def test_normal_stress_row_shell(m, c):
    """The traction-free Stokes shell, dot(r_hat, dot(strain, r_hat)) - c*p
    = 0 with the tangential stress rows at both walls: the row is linear
    in the IR, the tau layout is the stress-free shell's, each l >= 1 has
    two more finite eigenvalues than the impenetrable shell (one per
    wall), and the spectrum holds the toroidal family d/dr (T/r) = 0, the
    poloidal roots of _traction_free_roots for l = 1, 2 and the rigid
    rotation and translation at lambda = 0 (two each at m <= 1).  The
    l = 0 block holds one gauge-tau artefact, censused finite or not
    with its size: its poloidal solution 1/r^2 is outside the basis,
    unlike the ball's r and its exact 10/c."""
    tangential = [f"dot({hat}, {STRAIN}) = 0" for hat in ("theta_hat", "phi_hat")]

    def stokes(*bcs):
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_unknown("p", "scalar")
        pb.add_equation("dt(v) - lap(v) + grad(p) = 0")
        pb.add_equation("div(v) = 0")
        for at in ("inner", "outer"):
            for bc in bcs:
                pb.add_bc(bc, at=at)
        return pb

    targets = []
    for ell in (1, 2):
        targets.extend(k ** 2 for k in _toroidal_shell_roots(ell))
        targets.extend(k ** 2 for k in _traction_free_roots(ell, c))
    errs = []
    for grid in SHELL:
        ref = stokes("dot(r_hat, v) = 0", *tangential).compile(grid, m=m, backend="dedalus",
                                                                 gauge_unknowns=["p"])
        cps = {}
        for sign in ("-", "+", ""):
            term = f"{sign} {c}*p" if sign else ""
            cps[sign] = stokes(f"dot(r_hat, {STRAIN}) {term} = 0", *tangential).compile(
                grid, m=m, backend="dedalus", gauge_unknowns=["p"])
        cp = cps["-"]
        assert [t.name for t in cp.tau_fields] == ["tau_v1", "tau_v2", "tau_p_gauge"]
        scale = abs(cp.A_total).max()
        mean = (cps["-"].A_total + cps["+"].A_total) / 2
        assert abs(mean - cps[""].A_total).max() < 1e-13 * scale
        assert abs(cps["-"].A_total - cps[""].A_total).max() > 0.1 * c
        lam, n_inf = _census(cp, 1.0)
        extra = n_inf - _census(ref, 1.0)[1] + 2 * (grid.N_theta - m)
        assert extra in ((0, 1) if m == 0 else (0,)), extra
        assert np.count_nonzero(np.abs(lam) < 1e-8) == 2
        errs.append(_nearest_errors(lam, targets))
    _spectral(errs, 1e-6)


def test_theta_dependent_wall_coefficient_rejected_on_the_shell():
    pb = _helmholtz("dot(r_hat, grad(p)) + g0*p = 0", backgrounds={"g0": "1 + cos(theta)**2"})
    with pytest.raises(NotImplementedError, match="theta-dependent coefficient"):
        pb.compile(SHELL[0], m=1, backend="dedalus")


@pytest.mark.parametrize("rows", [
    ("dot(theta_hat, v) + p = 0", "dot(phi_hat, v) + p = 0"),
    ("dot(theta_hat, v) + dot(phi_hat, v) = 0",),
], ids=["scalar_term", "mixed"])
def test_mixed_tangential_row_rejected_on_the_shell(rows):
    """A tangential projection is a vector on the sphere and shares a row
    only with projections on the same unit vector, on the shell as on the
    ball."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_equation("dt(v) - lap(v) + grad(p) = 0")
    pb.add_equation("div(v) = 0")
    for at in ("inner", "outer"):
        pb.add_bc("dot(r_hat, v) = 0", at=at)
        for row in rows:
            pb.add_bc(row, at=at)
    with pytest.raises(NotImplementedError, match="same unit vector"):
        pb.compile(SHELL[0], m=1, backend="dedalus", gauge_unknowns=["p"])


def test_lone_tangential_composite_row_rejected_on_the_shell():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_parameter("ls", 0.5)
    pb.add_equation("dt(v) - lap(v) = 0")
    for at in ("inner", "outer"):
        pb.add_bc("dot(r_hat, v) = 0", at=at)
        pb.add_bc(f"dot(theta_hat, v) + ls*dot(theta_hat, {STRAIN}) = 0", at=at)
    with pytest.raises(NotImplementedError, match="phi_hat"):
        pb.compile(SHELL[0], m=1, backend="dedalus")
