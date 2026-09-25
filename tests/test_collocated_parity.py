"""The collocated core against the shared boundary and constraint
contracts (docs/ir_boundary_and_constraints.md): composite boundary
rows, tensor unknowns and definitions, divergence cleaning of a written
involution, and the vector potential.

Closed forms, all with the pencil convention K x = lambda M x and
lambda = -(growth rate):

  * Robin walls on the scalar Helmholtz problem, dot(r_hat, grad(p))
    + h p = 0: on the unit ball k j_l'(k) + h j_l(k) = 0, lambda = k^2;
    on the shell with the outward-normal Robin condition at both walls
    (dot(r_hat, grad(p)) - h p = 0 at r = r_i) the 2 x 2 Bessel
    determinant of a j_l(k r) + b y_l(k r);
  * Navier slip on toroidal diffusion, (1 - l_s) j_l(k) + l_s k j_l'(k)
    = 0, from a theta / phi pair of composite rows on the sigma = +1 and
    -1 wall rows;
  * the traction-free Stokes ball, dot(r_hat, dot(strain, r_hat)) - c p
    = 0: the toroidal family k j_l'(k) = j_l(k) and the poloidal roots
    of the 2 x 2 determinant of tests/test_dedalus_ball.py; the l = 0
    root lambda = 10 / c of that test is an artefact of the Dedalus
    gauge tau (div(v) + tau = 0 admits v = alpha r) and has no
    counterpart on a core that imposes div(v) = 0 at every node;
  * free magnetic decay in a perfectly conducting shell and ball, the b
    formulation with Div(b) = 0 written and no multiplier: the toroidal
    rates from d(r T)/dr = 0 and the poloidal ones from S = 0
    (tests/test_analytic_evp.py, benchmark 4); the vector potential gives
    the same rates;
  * the 2 kG dipole shift of the m = 8 sectoral Rossby mode of the Bekki
    Figure-2 shell in the b and A formulations (tests/test_staggered_mhd.py).

The orders are those of the order-4 stencils at the (12 x 18, 18 x 27)
pair, measured on the RMS error over the targets and asserted above the
floor of tests/test_analytic_evp.py.
"""

import sys

import numpy as np
import pytest
import scipy.sparse as sps
import sympy as sp
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

from spdee import Problem
from spdee.backends.fd2.ball_origin import tensor_origin_basis
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
from spdee.backends.fd2.layout import SPIN_COMPONENTS_TENSOR
from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.tensor_unknown_sparse import (
    build_r2_div_tensor_unk_block, build_r_tensor_of_vector_block,
    tensor_unknown_placeholders,
)
from spdee.ir import r, theta
from spdee.ir.tensor_reduce import (
    grad_vector_spin_components, r2_div_tensor_spin_components,
)
from spdee.solvers import dense

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analytic.manufactured import manufactured  # noqa: E402
from test_analytic_evp import _decay_targets, _radial_roots, decay_problem  # noqa: E402

RI = 0.5
RES = ((12, 18), (18, 27))
RATIO = 1.5
MIN_ORDER = 3.2
ORDER_FLOOR = 3e-4
STRAIN = "dot(grad(v) + trans(grad(v)), r_hat)"


def _grid(geometry, level):
    n_r, n_t = RES[level]
    if geometry == "ball":
        return make_ball_grid(N_r=n_r, N_theta=n_t, r_outer=1.0)
    return make_shell_grid(N_r=n_r, N_theta=n_t, r_inner=RI, r_outer=1.0)


def _roots(f, kmax=20.0, n=3, kmin=1e-3):
    ks = np.linspace(kmin, kmax, 6000)
    v = f(ks)
    out = []
    for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        out.append(brentq(f, ks[i], ks[i + 1]))
    return np.array(out[:n])


def _census(cp, sigma, resid=None):
    """The finite eigenvalues of a compiled pencil, gated on the pencil
    residual when ``resid`` is given."""
    K, M = cp.assemble_evp_matrices()
    out = dense(K, M, sigma=sigma, backend="auto", vectors=resid is not None,
                compute_resid=resid is not None)
    keep = np.isfinite(out.lam)
    if resid is not None:
        keep &= out.resid < resid
    return out.lam[keep]


def _nearest(lam, targets):
    return np.array([lam[np.argmin(np.abs(lam - t))] for t in targets])


def _errors(compile_at, targets, sigma, resid=None):
    """(2, n) relative errors of the nearest eigenvalues at the two
    resolutions, and the RMS order."""
    errs = []
    for level in (0, 1):
        lam = _census(compile_at(level), sigma, resid)
        errs.append(np.abs(_nearest(lam, targets) - targets) / np.abs(targets))
    errs = np.array(errs)
    rms = np.sqrt((errs ** 2).mean(axis=1))
    order = float(np.log(rms[0] / rms[1]) / np.log(RATIO))
    print(f"PARITY | rms error {rms[0]:.2e} -> {rms[1]:.2e} | order {order:.2f} | "
          f"max {errs[1].max():.2e}")
    return errs, order, rms


def _assert_order(order, rms, min_order=MIN_ORDER):
    if rms[0] >= ORDER_FLOOR:
        assert order > min_order, (order, rms)


# ---------------------------------------------------------------------------
# 1. Composite boundary rows
# ---------------------------------------------------------------------------

def _helmholtz(bc, params=None):
    pb = Problem()
    pb.add_unknown("p", "scalar")
    for name, value in (params or {}).items():
        pb.add_parameter(name, value)
    pb.add_equation("dt(p) - lap(p) = 0")
    for row, at in bc:
        pb.add_bc(row, at=at)
    return pb


def _robin_ball_targets(m, h, n_per_ell=3, max_ell=6):
    out = []
    for ell in range(m, max_ell + 1):
        ks = _roots(lambda k: k * spherical_jn(ell, k, derivative=True)
                    + h * spherical_jn(ell, k), n=n_per_ell)
        out.extend(ks ** 2)
    return np.sort(out)


def _robin_shell_targets(m, h, n_per_ell=2, max_ell=4):
    """p = a j_l(k r) + b y_l(k r) with k f'(k) + h f(k) = 0 at r = 1 and
    k f'(k r_i) - h f(k r_i) = 0 at r = r_i (the outward normal is -r_hat
    there, so both walls are dissipative)."""
    def det(k, ell):
        def wall(f, x, sign):
            return k * f(ell, x, derivative=True) + sign * h * f(ell, x)
        return (wall(spherical_jn, k, +1) * wall(spherical_yn, k * RI, -1)
                - wall(spherical_yn, k, +1) * wall(spherical_jn, k * RI, -1))
    out = []
    for ell in range(m, max_ell + 1):
        ks = _roots(lambda k: det(k, ell), kmax=25.0, n=n_per_ell, kmin=0.5)
        out.extend(ks ** 2)
    return np.sort(out)


@pytest.mark.parametrize("h", [1, 3])
@pytest.mark.parametrize("m", [0, 1, 2])
def test_helmholtz_robin_ball(m, h):
    """dot(r_hat, grad(p)) + h*p = 0 on the unit ball: lambda = k^2 with
    k j_l'(k) + h j_l(k) = 0 (h = 1, l = 0: k = pi/2, 3pi/2; h = 3,
    l = 0: 2.288930, 3.405608) at the FD order."""
    targets = _robin_ball_targets(m, h)[:6]
    pb = _helmholtz([("dot(r_hat, grad(p)) + h*p = 0", "outer")], {"h": float(h)})
    errs, order, rms = _errors(
        lambda lv: pb.compile(_grid("ball", lv), m=m, backend="fd2"), targets, -1.0)
    assert errs[1].max() < 3e-3, errs
    _assert_order(order, rms)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_helmholtz_robin_shell(m):
    """The shell analogue with the outward-normal Robin condition at both
    walls: the Bessel determinant of ``_robin_shell_targets``."""
    h = 2.0
    targets = _robin_shell_targets(m, h)[:5]
    pb = _helmholtz([("dot(r_hat, grad(p)) + h*p = 0", "outer"),
                     ("dot(r_hat, grad(p)) - h*p = 0", "inner")], {"h": h})
    errs, order, rms = _errors(
        lambda lv: pb.compile(_grid("shell", lv), m=m, backend="fd2"), targets, -1.0)
    assert errs[1].max() < 3e-3, errs
    _assert_order(order, rms)


def test_biot_wall_binds_the_parameters():
    """The radiative (Biot) wall on thermal diffusion with symbolic kappa
    and Bi: lambda = kappa k^2 with k j_l'(k) + Bi j_l(k) = 0, the
    coefficients bound into the composite row at compile time; every
    spelling of the coefficient gives the same pencil."""
    kappa, Bi = 0.1, 2.0
    targets = kappa * _robin_ball_targets(2, Bi)[:4]

    def problem(bc):
        pb = Problem()
        pb.add_unknown("S", "scalar")
        pb.add_parameter("kappa", kappa)
        pb.add_parameter("Bi", Bi)
        pb.add_equation("dt(S) - kappa*lap(S) = 0")
        pb.add_bc(bc, at="outer")
        return pb

    errs, order, rms = _errors(
        lambda lv: problem("dot(r_hat, grad(S)) + Bi*S = 0").compile(
            _grid("ball", lv), m=2, backend="fd2"), targets, -0.1)
    assert errs[1].max() < 3e-3, errs
    ref = problem("dot(r_hat, grad(S)) + Bi*S = 0").compile(_grid("ball", 0), m=2, backend="fd2")
    K_ref, _ = ref.assemble_evp_matrices()
    for bc in ("2*S + dot(grad(S), r_hat) = 0", "-(dot(r_hat, grad(S)) + Bi*S) = 0",
               "r**2*dot(r_hat, grad(S)) + (r**2 + 1)*S = 0"):
        K, _ = problem(bc).compile(_grid("ball", 0), m=2, backend="fd2").assemble_evp_matrices()
        assert abs(abs(K) - abs(K_ref)).max() < 1e-12 * abs(K_ref).max(), bc


def _slip_roots(ell, ls, n=2):
    return _roots(lambda k: (1 - ls) * spherical_jn(ell, k)
                  + ls * k * spherical_jn(ell, k, derivative=True), n=n)


def _toroidal_roots(ell, n=2):
    return _roots(lambda k: k * spherical_jn(ell, k, derivative=True)
                  - spherical_jn(ell, k), n=n)


def _traction_free_roots(ell, c, n=2):
    """The poloidal roots of the traction-free Stokes ball
    (tests/test_dedalus_ball.py)."""
    def det(k):
        f = spherical_jn(ell, k)
        f1 = k * spherical_jn(ell, k, derivative=True)
        f2 = -2 * f1 - (k ** 2 - ell * (ell + 1)) * f
        n_ = (2 * ell * (ell - 1) - c * k ** 2, 2 * ell * (ell + 1) * (f1 - f))
        t = (2 * (ell - 1), f2 + (ell * (ell + 1) - 2) * f)
        return n_[0] * t[1] - n_[1] * t[0]
    return _roots(det, n=n)


@pytest.mark.parametrize("m", [0, 1])
def test_navier_slip_ball(m):
    """dot(t_hat, v) + ls*dot(t_hat, dot(strain, r_hat)) = 0 for both
    tangents, two composite rows on the sigma = +1 and -1 wall rows of
    v: the toroidal rates Ek k^2 with (1 - l_s) j_l(k) + l_s k j_l'(k)
    = 0."""
    Ek, ls = 0.01, 0.5
    targets = np.array([Ek * k ** 2 for ell in range(max(m, 1), 4)
                        for k in _slip_roots(ell, ls)])

    def compile_at(level):
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_parameter("Ek", Ek)
        pb.add_parameter("ls", ls)
        pb.add_equation("dt(v) - Ek*lap(v) = 0")
        pb.add_bc("dot(r_hat, v) = 0", at="outer")
        for hat in ("theta_hat", "phi_hat"):
            pb.add_bc(f"dot({hat}, v) + ls*dot({hat}, {STRAIN}) = 0", at="outer")
        cp = pb.compile(_grid("ball", level), m=m, backend="fd2")
        hosts = {t.name: cp._bc_host(t)[1][1] for t in cp.bc_tasks}
        assert sorted(hosts.values()) == [-1, 0, 1]
        return cp

    errs, order, rms = _errors(compile_at, targets, -0.5)
    assert errs[1].max() < 2e-3, errs
    _assert_order(order, rms)


def _stokes(*bcs):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_equation("dt(v) - lap(v) + grad(p) = 0")
    pb.add_equation("div(v) = 0")
    for bc in bcs:
        pb.add_bc(bc, at="outer")
    return pb


@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("m", [0, 1])
def test_normal_stress_ball(m, c):
    """The traction-free Stokes ball: the normal-stress row constrains v
    and p on the sigma = 0 wall row of v, linearly in the coefficient of
    p; the spectrum holds the toroidal family k j_l'(k) = j_l(k) and the
    poloidal roots of the Bessel determinant (l = 2, c = 2: 8, 28.211083;
    l = 1: 14.978747)."""
    tangential = [f"dot({hat}, {STRAIN}) = 0" for hat in ("theta_hat", "phi_hat")]
    targets = []
    for ell in range(max(m, 1), 4):
        targets.extend(k ** 2 for k in _toroidal_roots(ell))
        targets.extend(k ** 2 for k in _traction_free_roots(ell, c))
    targets = np.array(targets)

    def compile_at(level, sign="-"):
        term = f"{sign} {c}*p" if sign else ""
        return _stokes(f"dot(r_hat, {STRAIN}) {term} = 0", *tangential).compile(
            _grid("ball", level), m=m, backend="fd2", gauge_unknowns=["p"])

    pencils = {sign: compile_at(0, sign).assemble_evp_matrices()[0] for sign in ("-", "+", "")}
    scale = abs(pencils["-"]).max()
    assert abs((pencils["-"] + pencils["+"]) / 2 - pencils[""]).max() < 1e-13 * scale
    D = pencils["-"] - pencils[""]
    assert D.nnz == _grid("ball", 0).N_theta and abs(abs(D).max() - c) < 1e-14
    cp = compile_at(0)
    row = next(t for t in cp.bc_tasks if "p" in t.metadata.get("unknowns", ()))
    assert cp._bc_host(row)[1][1] == 0 and row.unknown_name == "v"
    errs, order, rms = _errors(compile_at, targets, -0.5, resid=1e-8)
    assert errs[1].max() < 5e-3, errs
    _assert_order(order, rms)


# ---------------------------------------------------------------------------
# 2. Tensor unknowns and definitions
# ---------------------------------------------------------------------------

def viscous_problem(viscous, Ek=1e-2, dt_T=False, geometry="shell"):
    """dt(v) + 2 ez x v + grad(p) - Ek (viscous) = 0, div(v) = 0 with
    stress-free walls; a tensor unknown T when the viscous term uses it,
    defined by T - grad(v) - trans(grad(v)) = 0 (under Dt with ``dt_T``)."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    if "T" in viscous:
        pb.add_unknown("T", "tensor")
    pb.add_parameter("Ek", Ek, symbolic=False)
    pb.add_equation(f"dt(v) + 2*cross(ez, v) + grad(p) - Ek*({viscous}) = 0",
                    name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    if "T" in viscous:
        pb.add_equation(("dt(T) + " if dt_T else "") + "T - grad(v) - trans(grad(v)) = 0",
                        name="def")
    if geometry == "shell":
        pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    return pb


def _maxdiff(X, Y):
    D = X - Y
    return abs(D).max() if D.nnz else 0.0


def _spectrum_near(cp, sigma, radius=0.6):
    lam = _census(cp, sigma, resid=1e-8)
    return np.sort_complex(lam[np.abs(lam - sigma) < radius])


@pytest.mark.parametrize("geometry", ["shell", "ball"])
def test_tensor_definition_is_eliminated(geometry):
    """T - grad(v) - trans(grad(v)) = 0 classifies as a definition, T gets
    no DOFs, and the eliminated pencil is entrywise the strain-divergence
    pencil with the same spectrum to rounding."""
    grid = _grid(geometry, 0)
    m = 2
    cpT = viscous_problem("div(T)", geometry=geometry).compile(grid, m=m, backend="fd2")
    cpS = viscous_problem("div(grad(v) + trans(grad(v)))", geometry=geometry).compile(
        grid, m=m, backend="fd2")
    rec = cpT.matrices.classification
    assert rec.equations["def"]["class"] == "definition"
    assert rec.definitions == {"T": "def"} and rec.placement["T"] == "tensor"
    assert cpT.matrices.eliminated_tensors == {"T": "def"}
    assert cpT.matrices.layout.primary_names == ["v", "p"]
    A_T, A_S = cpT.matrices.A_total, cpS.matrices.A_total
    assert _maxdiff(A_T, A_S) <= 1e-14 * abs(A_S).max()
    assert _maxdiff(cpT.matrices.B_total, cpS.matrices.B_total) == 0.0
    sT, sS = _spectrum_near(cpT, -0.05 + 0.5j), _spectrum_near(cpS, -0.05 + 0.5j)
    assert len(sT) == len(sS) >= 2
    assert np.abs(sT - sS).max() <= 1e-12


@pytest.mark.parametrize("m", [0, 1, 2, 3])
@pytest.mark.parametrize("geometry", ["shell", "ball"])
def test_tensor_under_dt_keeps_nine_blocks(geometry, m):
    """With dt(T) the tensor stays in the state as nine component
    blocks: the layout has 13 blocks, the definition is an evolution
    equation, its mass is diag(r) on the nine blocks, and on the ball the
    origin rows of T reduce its nine i_r = 0 rings to the 3, 2, 1, 0
    origin DOFs of a rank-2 tensor at |m| = 0, 1, 2, >= 3."""
    grid = _grid(geometry, 0)
    N = grid.N_r * grid.N_theta
    cp = viscous_problem("div(T)", dt_T=True, geometry=geometry).compile(
        grid, m=m, backend="fd2", gauge_unknowns=["p"])
    assert cp.matrices.classification.equations["def"]["class"] == "evolution"
    assert cp.matrices.layout.primary_names == ["v", "p", "T"]
    assert cp.matrices.layout.size == 13 * N
    T = cp.matrices.layout.field_slice("T")
    B = cp.matrices.B_total.tocsr()[T, T]
    diag = np.tile(grid.r, 9 * grid.N_theta).astype(complex)
    if geometry == "ball":
        diag[::grid.N_r] = 1.0
    assert abs(B - sps.diags(diag, format="csr")).max() < 1e-14
    K, M = cp.assemble_evp_matrices()
    assert K.shape == (13 * N, 13 * N)
    if geometry == "ball":
        d = {0: 3, 1: 2, 2: 1, 3: 0}[m]
        assert len(tensor_origin_basis(m, grid.theta)) == d
        rows = cp.matrices.equation_row_slices["def"]
        origin = [rows.start + c * N + j * grid.N_r
                  for c in range(9) for j in range(grid.N_theta)]
        with_mass = [row for row in origin if M[row].nnz]
        assert len(with_mass) == d
    lam = _census(cp, -0.05 + 0.5j, resid=1e-8)
    assert lam.size > 100


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("geometry", ["shell", "ball"])
def test_tensor_blocks_on_manufactured_fields(geometry, m):
    """The tensor-of-vector block applied to the nodes of a Cartesian-
    smooth vector u gives r grad(u) at the FD order, and the divergence
    block applied to the nodes of grad(u) gives r^2 div(grad(u)); on the
    ball the i_r = 0 rows hold the r -> 0 limits (the unscaled operator)."""
    u = manufactured(geometry, m, "vector", k=1)
    u_spin = dict(zip((-1, 0, +1), (sp.sympify(x) for x in _spin_sympy(u))))
    T_cov = grad_vector_spin_components(u_spin, m)
    div = r2_div_tensor_spin_components({k: r * v for k, v in T_cov.items()}, m)
    errs = {"grad": [], "div": []}
    for level in (0, 1):
        grid = _grid(geometry, level)
        R, Th = grid.mesh()
        bundle = DerivativeBundle(grid, order_r=4, order_theta=4)
        ph = {s: sp.Function(f"u_sigma_{lab}")(r, theta)
              for s, lab in ((-1, "m"), (0, "0"), (+1, "p"))}
        rT = {k: sp.expand(r * v) for k, v in T_cov.items()}
        rT_ph = {k: sp.expand(r * v) for k, v in grad_vector_spin_components(ph, m).items()}
        G = build_r_tensor_of_vector_block(rT_ph, ph, bundle, m, "u")
        u_flat = np.concatenate([_nodes(u_spin[s], R, Th, grid).ravel(order="F")
                                 for s in (-1, 0, +1)])
        exact = np.concatenate([_lam(rT[k])(R, Th).ravel(order="F")
                                for k in SPIN_COMPONENTS_TENSOR])
        got = G @ u_flat
        errs["grad"].append(_interior_error(got, exact, grid, 9))
        if geometry == "ball":
            limit = np.concatenate([_at_origin(rT[k] / r, Th) for k in SPIN_COMPONENTS_TENSOR])
            assert _origin_error(got, limit, grid, 9, np.abs(exact).max()) < 2e-2, m
        D = build_r2_div_tensor_unk_block("T", bundle, m)
        T_flat = np.concatenate([_nodes(T_cov[k], R, Th, grid).ravel(order="F")
                                 for k in SPIN_COMPONENTS_TENSOR])
        exact = np.concatenate([_lam(sp.expand(div[s]))(R, Th).ravel(order="F")
                                for s in (-1, 0, +1)])
        got = D @ T_flat
        errs["div"].append(_interior_error(got, exact, grid, 3))
        if geometry == "ball":
            limit = np.concatenate([_at_origin(sp.expand(div[s]) / r ** 2, Th)
                                    for s in (-1, 0, +1)])
            assert _origin_error(got, limit, grid, 3, np.abs(exact).max()) < 5e-2, m
    for name, (e0, e1) in errs.items():
        order = np.log(e0 / e1) / np.log(RATIO)
        print(f"PARITY | tensor {name} {geometry} m={m} | {e0:.2e} -> {e1:.2e} | order {order:.2f}")
        assert e1 < 1e-2 and order > 3.0, (name, e0, e1)
    assert tensor_unknown_placeholders("T")[(0, +1)].func.__name__ == "T_sigma_0p"


def _spin_sympy(v):
    return ((v[1] - sp.I * v[2]) / sp.sqrt(2), v[0], (v[1] + sp.I * v[2]) / sp.sqrt(2))


def _lam(expr):
    f = sp.lambdify((r, theta), expr, modules="numpy")

    def g(R, Th):
        return np.broadcast_to(np.asarray(f(R, Th), dtype=complex), R.shape)
    return g


def _at_origin(expr, Th, eps=1e-4):
    """The r -> 0 limit of a regular field on the theta ring, evaluated
    at r = eps (the 1/r factors of the spin components cancel only in
    the limit, which SymPy does not always resolve with theta symbolic)."""
    return _lam(expr)(np.full_like(Th[:1, :], eps), Th[:1, :])[0]


def _nodes(expr, R, Th, grid):
    """Grid values of a field regular at the origin, its r = 0 ring taken
    as the r -> 0 limit."""
    values = _lam(expr)(R, Th).copy()
    if grid.is_ball:
        values[0, :] = _at_origin(expr, Th)
    return values


def _interior_error(got, exact, grid, blocks):
    N = grid.N_r * grid.N_theta
    mask = np.ones(N, dtype=bool).reshape(grid.N_r, grid.N_theta, order="F")
    mask[0, :] = False
    mask[-1, :] = False
    mask = np.tile(mask.ravel(order="F"), blocks)
    return np.abs(got - exact)[mask].max() / np.abs(exact).max()


def _origin_error(got, limit, grid, blocks, scale):
    """Error of the i_r = 0 rows against the r -> 0 limit, relative to
    the scale of the r^p-scaled operator on the grid."""
    N = grid.N_r * grid.N_theta
    idx = np.array([c * N + j * grid.N_r for c in range(blocks) for j in range(grid.N_theta)])
    return np.abs(got[idx] - limit).max() / scale


# ---------------------------------------------------------------------------
# 3. Divergence cleaning of a written involution
# ---------------------------------------------------------------------------

def ball_decay_problem(formulation):
    """Free decay in a perfectly conducting ball, the spellings of
    ``decay_problem`` on the outer wall alone."""
    pb = Problem()
    if formulation == "A":
        pb.add_unknown("A", "vector")
        pb.add_unknown("psi", "scalar")
        pb.add_equation("dt(A) - lap(A) - grad(psi) = 0", name="induction")
        pb.add_equation("div(A) = 0", name="coulomb")
        pb.add_bc("dot(theta_hat, A) = 0", at="outer")
        pb.add_bc("dot(phi_hat, A) = 0", at="outer")
        pb.add_bc("psi = 0", at="outer")
        return pb
    pb.add_unknown("b", "vector")
    pb.add_equation("dt(b) - lap(b) = 0", name="induction")
    if formulation == "b":
        pb.add_equation("div(b) = 0", name="solenoidal")
    pb.add_bc("dot(r_hat, b) = 0", at="outer")
    pb.add_bc("dot(theta_hat, curl(b)) = 0", at="outer")
    pb.add_bc("dot(phi_hat, curl(b)) = 0", at="outer")
    return pb


def _ball_decay_targets():
    """Toroidal l = 1 from d(r T)/dr = 0 at r = 1 (k = 2.743707,
    6.116764) and poloidal l = 1, 2 from S = 0 (k = 4.493409, 5.763459)."""
    tor = _radial_roots("rT", 1, 2)
    return np.array([tor[0] ** 2, _radial_roots("dirichlet", 1, 1)[0] ** 2,
                     _radial_roots("dirichlet", 2, 1)[0] ** 2, tor[1] ** 2])


def _decay(geometry, formulation):
    problem = decay_problem if geometry == "shell" else ball_decay_problem
    targets = _decay_targets() if geometry == "shell" else _ball_decay_targets()
    compiled = {}

    def compile_at(level):
        compiled[level] = problem(formulation).compile(_grid(geometry, level), m=0,
                                                       backend="fd2")
        return compiled[level]

    errs, order, rms = _errors(compile_at, targets, float(np.mean(targets)), resid=1e-8)
    return errs, order, rms, compiled, targets


@pytest.mark.parametrize("geometry", ["shell", "ball"])
def test_free_decay_b_formulation(geometry):
    """Div(b) = 0 written with no multiplier: the classifier's divergence
    cleaning introduces __div_b, hosts the constraint on its rows and pins
    it at the walls (on a ball at m = 0 also by the mean-zero row, the pin
    of the collocated checkerboard in the multiplier); the decay rates match the
    Bessel conditions and the eigenvectors are solenoidal: the interior
    constraint rows to rounding, the rows the multiplier's Dirichlet walls,
    the origin ties and the gauge replace to the discretisation error."""
    errs, order, rms, compiled, targets = _decay(geometry, "b")
    assert errs[1].max() < 3e-3, errs
    _assert_order(order, rms, min_order=MIN_ORDER if geometry == "shell" else 2.5)
    cp = compiled[1]
    cm = cp.matrices
    assert cm.layout.primary_names == ["b", "__div_b"]
    rec = cm.classification
    assert rec.constraints == {"b": "involution"} and rec.involution_treatment == "saddle_point"
    cleaning = rec.divergence_cleaning["b"]
    assert (cleaning.multiplier, cleaning.constraint, cleaning.gauge) == (
        "__div_b", "solenoidal", "integral")
    assert cm.gauge_unknowns == (("__div_b",) if geometry == "ball" else ())
    assert [t.unknown_name for t in cm.implicit_bc_tasks] == ["__div_b"] * len(cleaning.walls)
    assert cm.paired_equation_of_unknown["__div_b"] == "solenoidal"
    wall_residual = []
    for level in (0, 1):
        cp = compiled[level]
        K, M = cp.assemble_evp_matrices()
        out = dense(K, M, sigma=float(np.mean(targets)), backend="auto", vectors=True,
                    keep_vectors=True)
        keep = np.isfinite(out.lam) & (out.resid < 1e-8)
        lam, V = out.lam[keep], out.V[:, keep]
        rows = cp.matrices.equation_row_slices["solenoidal"]
        div_rows = cp.matrices.A_total[rows, :]
        N_r = cp.matrices.grid.N_r
        interior = np.ones(rows.stop - rows.start, dtype=bool)
        interior[N_r - 1::N_r] = False
        interior[::N_r] = False
        interior[cp._gauge_host("__div_b")[0] - rows.start] = False
        worst = 0.0
        for t in targets:
            x = V[:, np.argmin(np.abs(lam - t))]
            res = np.abs(div_rows @ x) / np.abs(x).max()
            assert res[interior].max() < 1e-8, t
            worst = max(worst, res[~interior].max())
        wall_residual.append(worst)
    print(f"PARITY | div b at the walls {wall_residual[0]:.2e} -> {wall_residual[1]:.2e}")
    assert wall_residual[1] < wall_residual[0] / 2


@pytest.mark.parametrize("geometry", ["shell", "ball"])
def test_free_decay_conducting_wall_alone(geometry):
    """Without the constraint the conducting wall closes the vector
    Laplacian by itself (the tangential curl(b) rows), as on the spectral
    cores; the rates are the same."""
    errs, order, rms, compiled, _ = _decay(geometry, "b_free")
    assert errs[1].max() < 3e-3, errs
    _assert_order(order, rms, min_order=MIN_ORDER if geometry == "shell" else 2.5)
    assert compiled[1].matrices.layout.primary_names == ["b"]
    kinds = {t.name: t.kind for t in compiled[1].bc_tasks}
    assert sum(k == "projection_curl_cross" for k in kinds.values()) == (
        4 if geometry == "shell" else 2)


# ---------------------------------------------------------------------------
# 4. Vector potential
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("geometry", ["shell", "ball"])
def test_free_decay_vector_potential(geometry):
    """The A formulation (Coulomb gauge as a saddle point on psi) gives the
    b formulation's rates and the closed forms."""
    errs, order, rms, compiled, targets = _decay(geometry, "A")
    assert errs[1].max() < 1e-2, errs
    _assert_order(order, rms, min_order=MIN_ORDER if geometry == "shell" else 1.5)
    lam_A = _census(compiled[1], float(np.mean(targets)), resid=1e-8)
    lam_b = _census(_decay(geometry, "b")[3][1], float(np.mean(targets)), resid=1e-8)
    gap = np.abs(_nearest(lam_A, targets) - _nearest(lam_b, targets)) / targets
    assert gap.max() < 1e-2, gap


def test_dipole_rossby_shift_b_and_A_formulations():
    """The 2 kG dipole on the m = 8 Bekki Figure-2 shell: the b
    formulation (conducting walls, the tangential electric field rows
    eta curl(b) - cross(v, B0) = 0) and the vector potential (the Lorentz
    force cross(curl(curl(A)), B0), Coulomb gauge, tangential A = 0 and
    psi = 0) shift the sectoral Rossby mode alike as the grid is refined,
    and each lands near the Dedalus value of its own formulation
    (tests/test_staggered_mhd.py: -0.0676-0.0091j for b, -0.0686-0.0102j
    for A)."""
    import test_staggered_mhd as ts

    def compile_at(system, n_r, n_t):
        grid = make_shell_grid(N_r=n_r, N_theta=n_t, r_inner=ts.R_LO, r_outer=ts.R_HI)
        return compile_problem(system, grid, m=ts.M, order_r=4, order_theta=4)

    diffs, shifts = [], {}
    for n_r, n_t in ((12, 24), (16, 32)):
        K_h, M_h = compile_at(ts.hydro_system(), n_r, n_t).assemble_evp_matrices()
        om0 = ts.rossby(K_h, M_h)
        assert abs(om0 - (-0.2206 - 0.0193j)) < 2e-3, om0
        out = {}
        for tag, system in (("b", ts.mhd_system(2000.0)[0]),
                            ("A", ts.potential_system(2000.0)[0])):
            cp = compile_at(system, n_r, n_t)
            names = cp.matrices.layout.primary_names
            assert names == ["v", "rho1", "s1", "b"] + (["psi"] if tag == "A" else [])
            K, M = cp.assemble_evp_matrices()
            out[tag] = ts.rossby(K, M, omega_guess=om0) - om0
        diffs.append(abs(out["b"] - out["A"]))
        shifts[(n_r, n_t)] = out
        print(f"PARITY | dipole shift {n_r}x{n_t} | b {out['b']:.5f} | A {out['A']:.5f}")
    assert diffs[1] < diffs[0], diffs
    fine = shifts[(16, 32)]
    assert abs(fine["b"] - (-0.0676 - 0.0091j)) < 5e-3, fine
    assert abs(fine["A"] - (-0.0686 - 0.0102j)) < 5e-3, fine
    assert diffs[1] < 0.05 * abs(fine["A"]), diffs
