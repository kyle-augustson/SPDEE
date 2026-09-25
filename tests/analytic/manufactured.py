"""Manufactured-solution boundary-value problems for one IR operator.

A case names an operator as a STRING, so the same problem statement is
compiled by every core. The string is evaluated twice: once in the IR
namespace of :class:`spdee.Problem`, giving the equations the core
compiles, and once in the symbolic namespace of
``analytic.vector_calculus``, giving the exact image of the manufactured
solution. Nothing about the operator is written down twice.

The boundary-value problem is ``Op(u) = f`` with ``f`` the symbolic
image of the manufactured ``u``. Most of the vocabulary is not an
elliptic second-order operator of its own unknown — a first-order
operator, a pointwise product, an operator that changes rank — so the
problem actually posed is ``Lap(u) + Op(u) = f``: the Laplacian carrier
makes the system square and elliptic while a wrong ``Op`` still moves
the solution at O(1), which collapses the measured order. The cases
that are elliptic on their own (Lap and its coefficient variants) carry
no Laplacian and are the bare ``Op(u) = f``. When ``Op`` changes rank
the probe gets its own unknown ``w`` of the output rank,

    Lap(u) = f_u,   Lap(w) + Op(u) = Op(u_manufactured),

whose exact solution is ``w = 0``: the probe unknown then holds the
operator's discretization error directly.

The manufactured fields vanish on every wall, so the boundary rows are
homogeneous on every core and no core-specific boundary-value
convention enters. They are Cartesian-analytic (built from ``(x+iy)^m``,
``z`` and ``r^2``), so the ball origin treatments apply.

How each core is driven
-----------------------

collocated (``spdee.backends.fd2.compiled_problem.compile_problem``)
    Its own BVP path: the source enters as a background field of the
    equation and ``solve_bvp`` installs origin regularity, boundary rows
    and gauge rows and solves ``A x = b``.

staggered (``discretization='staggered'``)
    The general C-grid assembler refuses algebraic equations, so it has
    no source-driven BVP path; ``b_total`` is also neither
    r-premultiplied nor reduced alongside ``A_total``. The equations are
    therefore written in evolution form and the source is applied
    through the mass matrix (below).

sh_fd, dedalus
    Neither compiles a source term: ``sh_fd`` has no RHS vector at all
    and the Dedalus backend builds a homogeneous ``d3.EVP``. They are
    driven through the mass matrix as well.

The mass-matrix drive. An equation written as ``Dt(u) - L(u) = 0``
compiles to ``K = -mass * L`` and ``M = mass`` on that unknown, with the
core's own row mass and row scaling in both. Solving

    K x = -M x_f,     x_f the source in the core's own state layout,

is therefore exactly ``L(u) = f`` in that core's discretization,
whatever the row scaling is; boundary rows carry no ``Dt`` so their mass
row is empty and they stay homogeneous.
"""

import functools

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import sympy as sp

from spdee import Problem
from spdee.ir import r, theta

from . import vector_calculus as vc

R_INNER = 0.35
EK = 0.37

GEOMETRIES = ("shell", "ball")


# ---------------------------------------------------------------------------
# Manufactured fields and backgrounds
# ---------------------------------------------------------------------------

def _wall_factor(geometry):
    """A Cartesian-analytic factor vanishing on every wall."""
    if geometry == "ball":
        return 1 - r ** 2
    return (r - R_INNER) * (1 - r)


def _regular(m, k):
    """A Cartesian-analytic scalar of azimuthal order m: (x+iy)^m g(z, r^2)."""
    z = r * sp.cos(theta)
    g = [sp.cos(z) + r ** 2 / 3, sp.sin(z + sp.Rational(3, 10) * r ** 2)][k]
    return (r * sp.sin(theta)) ** m * g


@functools.lru_cache(maxsize=None)
def manufactured(geometry, m, kind, k=0):
    """The manufactured field of the given rank; zero on every wall."""
    w = _wall_factor(geometry)
    if kind == "scalar":
        return w * _regular(m, k)
    inner = vc.Vec(r * _regular(m, 1 - k), 0, 0)
    return (vc.grad(_regular(m, k), m) + vc.curl(inner, m)) * w


@functools.lru_cache(maxsize=None)
def backgrounds(m):
    """Cartesian-analytic backgrounds: a vector, a scalar and a tensor."""
    z, s = r * sp.cos(theta), sp.sin(theta)
    B0 = vc.Vec(sp.cos(theta) * (1 + r ** 2 / 4) + r / 4,
                -sp.sin(theta) * (1 + r ** 2 / 4), r * s / 2)
    c0 = 1 + r ** 2 / 3 + z / 5
    T0 = vc.Ten([[c0 + z, -r * s, 0], [0, c0, 0], [0, 0, c0]])
    return {"B0": B0, "c0": c0, "T0": T0, "Ek": sp.Symbol("Ek")}


# ---------------------------------------------------------------------------
# Case table
# ---------------------------------------------------------------------------

class Case:
    """One operator, as the string that both namespaces evaluate.

    ``unknown`` is the rank of the field the operator acts on and
    ``output`` the rank of its image; they differ exactly when the case
    needs a separate probe unknown. ``carrier`` adds the Laplacian that
    makes the operator's own equation square and elliptic; the operators
    that are already an elliptic second-order operator of their own
    unknown set it False and are compiled as the bare ``Op(u) = f``.
    """

    __slots__ = ("name", "probe", "unknown", "output", "carrier", "note")

    def __init__(self, name, probe, unknown, output, carrier=True, note=""):
        self.name, self.probe = name, probe
        self.unknown, self.output = unknown, output
        self.carrier, self.note = carrier, note

    @property
    def two_field(self):
        return self.unknown != self.output

    def __repr__(self):
        return f"Case({self.name!r})"


CASES = (
    Case("Lap(scalar)", "lap(u)", "scalar", "scalar", carrier=False),
    Case("LapH(scalar)", "laph(u)", "scalar", "scalar"),
    Case("Lap(vector)", "lap(u)", "vector", "vector", carrier=False),
    Case("Grad(scalar)", "grad(u)", "scalar", "vector"),
    Case("Div(vector)", "div(u)", "vector", "scalar"),
    Case("Curl(vector)", "curl(u)", "vector", "vector"),
    Case("Curl(Curl(vector))", "-curl(curl(u))", "vector", "vector",
         note="negated: Lap + CurlCurl = Grad Div is singular"),
    Case("Grad(Div(vector))", "grad(div(u))", "vector", "vector"),
    Case("Div(strain)", "div(grad(u) + trans(grad(u)))", "vector", "vector"),
    Case("Div(Transpose(Grad))", "div(trans(grad(u)))", "vector", "vector"),
    Case("Dot(B0, Grad(vector))", "dot(B0, grad(u))", "vector", "vector"),
    Case("Dot(B0, T(vector))", "dot(B0, grad(u) + trans(grad(u)))", "vector", "vector"),
    Case("Dot(T(vector), B0)", "dot(grad(u) + trans(grad(u)), B0)", "vector", "vector"),
    Case("Dot(B0, Grad(scalar))", "dot(B0, grad(u))", "scalar", "scalar"),
    Case("Dot(r_hat, vector)", "dot(r_hat, u)", "vector", "scalar"),
    Case("Dot(B0, vector)", "dot(B0, u)", "vector", "scalar"),
    Case("Dot(T0, vector)", "dot(T0, u)", "vector", "vector"),
    Case("Dot(vector, T0)", "dot(u, T0)", "vector", "vector"),
    Case("Cross(B0, vector)", "cross(B0, u)", "vector", "vector"),
    Case("Cross(vector, B0)", "cross(u, B0)", "vector", "vector"),
    Case("Cross(Curl(vector), B0)", "cross(curl(u), B0)", "vector", "vector"),
    Case("Curl(Cross(vector, B0))", "curl(cross(u, B0))", "vector", "vector"),
    Case("B0 * scalar", "B0 * u", "scalar", "vector"),
    Case("B0 * Div(vector)", "B0 * div(u)", "vector", "vector"),
    Case("c0 * Lap(vector)", "c0 * lap(u)", "vector", "vector", carrier=False),
    Case("Ek * Lap(scalar)", "Ek * lap(u)", "scalar", "scalar", carrier=False),
)

CASES_BY_NAME = {c.name: c for c in CASES}


# ---------------------------------------------------------------------------
# Symbolic images
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _image(case, geometry, m):
    """(u_manufactured, source of the u equation, source of the probe)."""
    ns = vc.namespace(m, backgrounds(m))
    u = manufactured(geometry, m, case.unknown, k=0)
    ns["u"] = u
    ns["Ek"] = sp.Float(EK)
    probe = eval(case.probe, {"__builtins__": {}}, ns)  # noqa: S307
    if case.two_field:
        return u, vc.lap(u, m), probe
    if not case.carrier:
        return u, probe, None
    lap_u = vc.lap(u, m)
    total = lap_u + probe if not isinstance(probe, vc.Vec) else vc.Vec(
        lap_u[i] + probe[i] for i in vc.AXES)
    return u, total, None


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def _add_source(pb, name, value):
    """A source background as numeric components.

    The operator images carry 1/r and 1/sin(theta) factors that cancel
    only across terms, so they are handed to the core as callables that
    take the limit at a singular node rather than as SymPy expressions
    the core would lambdify into a 0/0.
    """
    if isinstance(value, vc.Vec):
        return pb.add_background(name, v_r=_lam(value[0]), v_theta=_lam(value[1]),
                                 v_phi=_lam(value[2]))
    return pb.add_background(name, expr=_lam(value))


def _add_backgrounds(pb, m):
    bg = backgrounds(m)
    pb.add_background("B0", v_r=bg["B0"][0], v_theta=bg["B0"][1], v_phi=bg["B0"][2])
    pb.add_background("c0", expr=bg["c0"])
    names = ("r", "theta", "phi")
    pb.add_background("T0", components={
        (names[i], names[j]): bg["T0"][i][j] for i in vc.AXES for j in vc.AXES})
    pb.add_parameter("Ek", EK)


def _walls(geometry):
    return ("outer",) if geometry == "ball" else ("outer", "inner")


def _add_bcs(pb, name, kind, geometry):
    for at in _walls(geometry):
        if kind == "scalar":
            pb.add_bc(f"{name} = 0", at=at)
        else:
            for unit in ("r_hat", "theta_hat", "phi_hat"):
                pb.add_bc(f"dot({unit}, {name}) = 0", at=at)


def build_problem(case, geometry, m, form):
    """The Problem for one case; ``form`` is 'source' or 'evolution'.

    'source' writes ``Lap(u) + Op(u) - f = 0`` for a core that compiles a
    source term; 'evolution' writes ``Dt(u) - Lap(u) - Op(u) = 0``, whose
    source is applied through the mass matrix.
    """
    _u, f_u, f_w = _image(case, geometry, m)
    pb = Problem()
    pb.add_unknown("u", case.unknown)
    if case.two_field:
        pb.add_unknown("w", case.output)
    _add_backgrounds(pb, m)
    if form == "source":
        _add_source(pb, "f_u", f_u)
        if case.two_field:
            _add_source(pb, "f_w", f_w)
            pb.add_equation("lap(u) - f_u = 0", name="carrier")
            pb.add_equation(f"lap(w) + ({case.probe}) - f_w = 0", name="probe")
        elif case.carrier:
            pb.add_equation(f"lap(u) + ({case.probe}) - f_u = 0", name="carrier")
        else:
            pb.add_equation(f"({case.probe}) - f_u = 0", name="carrier")
    elif case.two_field:
        pb.add_equation("dt(u) - lap(u) = 0", name="carrier")
        pb.add_equation(f"dt(w) - lap(w) - ({case.probe}) = 0", name="probe")
    elif case.carrier:
        pb.add_equation(f"dt(u) - lap(u) - ({case.probe}) = 0", name="carrier")
    else:
        pb.add_equation(f"dt(u) - ({case.probe}) = 0", name="carrier")
    _add_bcs(pb, "u", case.unknown, geometry)
    if case.two_field:
        _add_bcs(pb, "w", case.output, geometry)
    return pb, f_u, f_w


# ---------------------------------------------------------------------------
# Numerical evaluation of the manufactured fields
# ---------------------------------------------------------------------------

_LIMIT_EPS = "1e-12"
_LIMIT_DPS = 60


@functools.lru_cache(maxsize=None)
def _lam(expr):
    """Grid evaluation of a symbolic field.

    The operator images carry 1/r and 1/sin(theta) factors that cancel
    only across terms, so a node at r = 0 (the collocated ball grid) or
    at a pole (the staggered theta edges) evaluates to 0/0 in double
    precision. Those nodes are taken as the limit, evaluated ``_LIMIT_EPS``
    away in ``_LIMIT_DPS``-digit arithmetic: the cancelling terms reach
    1/eps^2, which leaves about half the digits, and what is left is an
    O(eps) truncation.
    """
    fn = sp.lambdify((r, theta), expr, modules="numpy")
    cache = {}

    def call(R, T):
        R, T = np.asarray(R, dtype=float), np.asarray(T, dtype=float)
        with np.errstate(all="ignore"):
            out = np.asarray(fn(R, T), dtype=complex) + np.zeros_like(R, dtype=complex)
        bad = np.nonzero(~np.isfinite(out))
        if bad[0].size:
            import mpmath
            if "fn" not in cache:
                cache["fn"] = sp.lambdify((r, theta), expr, modules="mpmath")
            mpmath.mp.dps = _LIMIT_DPS
            eps = mpmath.mpf(_LIMIT_EPS)
            for idx in zip(*bad):
                rr = eps if R[idx] == 0.0 else mpmath.mpf(float(R[idx]))
                tt = mpmath.mpf(float(T[idx]))
                if float(np.sin(T[idx])) == 0.0:
                    tt = eps if float(T[idx]) < 1.0 else mpmath.pi - eps
                out[idx] = complex(cache["fn"](rr, tt))
        return out
    return call


def evaluate(value, R, T):
    """Grid values of a symbolic scalar (array) or Vec (3, ...) array."""
    if isinstance(value, vc.Vec):
        return np.stack([_lam(c)(R, T) for c in value.c])
    return _lam(value)(R, T)


def spin_from_physical(v):
    """(v^-1, v^0, v^+1) from physical (v_r, v_theta, v_phi)."""
    return ((v[1] - 1j * v[2]) / np.sqrt(2), v[0], (v[1] + 1j * v[2]) / np.sqrt(2))


def physical_from_spin(vm, v0, vp):
    return (v0, (vp + vm) / np.sqrt(2), 1j * (vm - vp) / np.sqrt(2))


class Result:
    """Errors of one solve, relative to the manufactured amplitude.

    ``error`` is the number the operator is judged on: the probe
    unknown's max norm when the case carries one (its exact value is
    zero, so it holds the operator's discretization error), otherwise
    the carrier's own solution error.
    """

    __slots__ = ("error", "error_u", "error_probe", "scale")

    def __init__(self, error_u, error_probe, scale, two_field):
        self.error_u, self.error_probe, self.scale = error_u, error_probe, scale
        self.error = error_probe if two_field else error_u

    def __repr__(self):
        return f"Result(error={self.error:.3e})"


def order_of(err_coarse, err_fine, n_coarse, n_fine):
    if not (err_coarse > 0 and err_fine > 0):
        return float("nan")
    return float(np.log(err_coarse / err_fine) / np.log(n_fine / n_coarse))


# ---------------------------------------------------------------------------
# Core drivers
# ---------------------------------------------------------------------------

def _fd_grid(geometry, n, aspect=1):
    from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
    nt = aspect * n
    if geometry == "ball":
        return make_ball_grid(N_r=n, N_theta=nt, r_outer=1.0)
    return make_shell_grid(N_r=n, N_theta=nt, r_inner=R_INNER, r_outer=1.0)


def _err(num, exact):
    """Max-norm difference; a non-finite solve reports infinity rather
    than losing to the NaN comparison of ``max``."""
    e = float(np.max(np.abs(np.asarray(num) - np.asarray(exact))))
    return e if np.isfinite(e) else float("inf")


def _run_collocated(case, geometry, m, n, order=4, aspect=1):
    pb, _f_u, _f_w = build_problem(case, geometry, m, "source")
    grid = _fd_grid(geometry, n, aspect)
    cp = pb.compile(grid, m=m, backend="fd2", order_r=order, order_theta=order)
    res = cp.solve_bvp()
    R, T = grid.mesh()
    u_ex = evaluate(manufactured(geometry, m, case.unknown), R, T)
    scale = float(np.max(np.abs(u_ex)))
    if case.unknown == "scalar":
        err_u = _err(res["u"], u_ex)
    else:
        got = physical_from_spin(res["u"][-1], res["u"][0], res["u"][+1])
        err_u = max(_err(got[i], u_ex[i]) for i in vc.AXES)
    err_w = 0.0
    if case.two_field:
        if case.output == "scalar":
            err_w = _err(res["w"], 0.0)
        else:
            err_w = max(_err(res["w"][s], 0.0) for s in (-1, 0, 1))
    return Result(err_u / scale, err_w / scale, scale, case.two_field)


def _mass_rhs_solve(K, M, x_f):
    """The BVP solve of the mass-matrix drive."""
    return spla.spsolve(K.tocsr(), -(M @ x_f))


# ---------------------------------------------------------------------------
# sh_fd: spectral state vectors
# ---------------------------------------------------------------------------

SH_ELL_MAX = 10


def _sh_grid(geometry, n):
    from spdee.backends.sh_fd import ShFdGrid
    r_in = 0.0 if geometry == "ball" else R_INNER
    return ShFdGrid(N_r=n, ell_max=SH_ELL_MAX, r_inner=r_in, r_outer=1.0, fd_order=4)


def _sh_quadrature(cp):
    """Gauss-Legendre theta nodes and the synthesis matrices per spin."""
    from spdee.backends.sh_fd import sYlm_theta
    nq = 2 * len(cp.ls) + 8
    x, w = np.polynomial.legendre.leggauss(nq)
    th = np.arccos(x)
    Y = {s: np.array([sYlm_theta(s, l, cp.m, th) for l in cp.ls]).T for s in (-1, 0, 1)}
    return th, w, Y


def _sh_analysis(Y, w, values):
    """Least-squares expansion of (nq, Nr) grid values on the harmonics."""
    root = np.sqrt(w)[:, None]
    return np.linalg.lstsq(root * Y, root * values, rcond=None)[0]


def _sh_state(cp, sources, th, w, Y):
    """The source in the sh_fd state layout (component order -, +, r)."""
    x = np.zeros(cp.N, dtype=complex)
    R = np.broadcast_to(cp.r[None, :], (len(th), cp.Nr))
    T = np.broadcast_to(th[:, None], (len(th), cp.Nr))
    for name, kind, off, size in cp.layout:
        value = sources.get(name)
        if value is None:
            continue
        if kind == "scalar":
            x[off:off + size] = _sh_analysis(Y[0], w, evaluate(value, R, T)).ravel()
            continue
        spin = dict(zip((-1, 0, 1), spin_from_physical(evaluate(value, R, T))))
        block = np.stack([_sh_analysis(Y[s], w, spin[s]) for s in (-1, 1, 0)])
        x[off:off + size] = block.ravel()
    return x


def _sh_fields(cp, x, th, Y):
    """Synthesized (nq, Nr) grid values per unknown, physical components."""
    out = {}
    for name, block in cp.unpack(x).items():
        kind = cp._kind[name]
        if kind == "scalar":
            out[name] = Y[0] @ block[0]
        else:
            spin = [Y[s] @ block[k] for k, s in enumerate((-1, 1, 0))]
            out[name] = np.stack(physical_from_spin(spin[0], spin[2], spin[1]))
    return out


def _run_sh_fd(case, geometry, m, n):
    from spdee.backends.sh_fd import compile_sh_fd
    pb, f_u, f_w = build_problem(case, geometry, m, "evolution")
    cp = compile_sh_fd(pb.build(), _sh_grid(geometry, n), m, params={sp.Symbol("Ek"): EK})
    th, w, Y = _sh_quadrature(cp)
    sources = {"u": f_u} if not case.two_field else {"u": f_u, "w": f_w}
    K, M = cp.assemble_evp_matrices()
    x = _mass_rhs_solve(K, M, _sh_state(cp, sources, th, w, Y))
    got = _sh_fields(cp, x, th, Y)
    R = np.broadcast_to(cp.r[None, :], (len(th), cp.Nr))
    T = np.broadcast_to(th[:, None], (len(th), cp.Nr))
    u_ex = evaluate(manufactured(geometry, m, case.unknown), R, T)
    scale = float(np.max(np.abs(u_ex)))
    err_u = _err(got["u"], u_ex)
    err_w = _err(got["w"], 0.0) if case.two_field else 0.0
    return Result(err_u / scale, err_w / scale, scale, case.two_field)


# ---------------------------------------------------------------------------
# dedalus: coefficient-space state vectors
# ---------------------------------------------------------------------------

def _dedalus_grid(geometry, n):
    from spdee.backends.dedalus import DedalusGrid
    r_in = 0.0 if geometry == "ball" else R_INNER
    return DedalusGrid(N_r=n, N_theta=n, r_inner=r_in, r_outer=1.0)


def _dedalus_kept_columns(cp):
    """Pencil columns that survive the auxiliary elimination.

    ``DedalusCompiledProblem._assemble`` drops the columns of its
    stand-in auxiliary fields and rebuilds them from the columns they
    reference; the reduced state is the rest, in order.
    """
    from spdee.backends.dedalus.compiler import _selected
    if cp._T is None:
        return None
    vars_ = cp.solver.problem.LHS_variables
    col_var = []
    for q, _ss in cp._blocks:
        sizes = np.cumsum([0] + [q.field_size(v) for v in vars_])
        col_var.append(np.searchsorted(sizes, _selected(q.pre_right_pinv), side="right") - 1)
    col_var = np.concatenate(col_var)
    aux = [i for i, v in enumerate(vars_) if any(v is t.field for t in cp._aux)]
    return np.nonzero(~np.isin(col_var, aux))[0]


def _dedalus_grid_values(cp, value):
    """A symbolic field on the Dedalus grid, with its exp(i m phi) factor."""
    phi = cp.dist.local_grids(cp.basis)[0]
    wave = np.exp(1j * cp.m * phi)
    vals = evaluate(value, cp._R, cp._TH)
    if isinstance(value, vc.Vec):
        return np.stack([vals[2] * wave, vals[1] * wave, vals[0] * wave])
    return vals * wave


def _dedalus_state(cp, sources):
    for f in cp.solver.state:
        f["c"] = 0
    for name, value in sources.items():
        cp.fields[name].change_scales(1)
        cp.fields[name]["g"] = _dedalus_grid_values(cp, value)
    for f in cp.solver.state:
        f["c"]
    x = np.concatenate([q.pre_right_pinv @ ss.gather(cp.solver.state)
                        for q, ss in cp._blocks])
    keep = _dedalus_kept_columns(cp)
    return x if keep is None else x[keep]


def _run_dedalus(case, geometry, m, n):
    from spdee.backends.dedalus import compile_dedalus
    pb, f_u, f_w = build_problem(case, geometry, m, "evolution")
    cp = compile_dedalus(pb.build(), _dedalus_grid(geometry, n), m,
                         params={sp.Symbol("Ek"): EK})
    sources = {"u": f_u} if not case.two_field else {"u": f_u, "w": f_w}
    K, M = cp.assemble_evp_matrices()
    x = _mass_rhs_solve(K, M, _dedalus_state(cp, sources))
    got = cp.eigenvector_fields(x)
    R, T = cp._R[0], cp._TH[0]
    u_ex = evaluate(manufactured(geometry, m, case.unknown), R, T)
    scale = float(np.max(np.abs(u_ex)))
    if case.unknown == "scalar":
        err_u = _err(got["u"], u_ex)
    else:
        err_u = max(_err(got["u"][2 - i], u_ex[i]) for i in vc.AXES)
    err_w = 0.0
    if case.two_field:
        err_w = _err(got["w"], 0.0)
    return Result(err_u / scale, err_w / scale, scale, case.two_field)


# ---------------------------------------------------------------------------
# staggered: C-grid state vectors
# ---------------------------------------------------------------------------

_STAG_COMPONENT = {"_r": 0, "_t": 1, "_p": 2}


def _stag_sub_grid(grids, location):
    return {"center": (grids.r_center, grids.theta_center),
            "r_edge": (grids.r_edge, grids.theta_center),
            "theta_edge": (grids.r_center, grids.theta_edge),
            "corner": (grids.r_edge, grids.theta_edge)}[location]


def _stag_nodes(field, grids, geometry, m):
    """The (r, theta) of a field's degrees of freedom, theta fastest.

    A face unknown's radial component loses its wall rings to the
    symmetric DOF reduction of ``_reduce_velocity_dofs``; on a ball at
    |m| <= 1 the origin ring is folded onto its (i_r = 0, j = 0) master
    with the cos/sin profile, which is exactly the origin structure of a
    regular vector, so evaluating the source on the surviving node is
    the right Galerkin value.
    """
    rv, tv = _stag_sub_grid(grids, field.location)
    R, T = np.repeat(rv, len(tv)), np.tile(tv, len(rv))
    if field.size == R.size:
        return R, T
    nt, nr = len(tv), len(rv)
    drop = set(range((nr - 1) * nt, nr * nt))
    if geometry == "shell":
        drop |= set(range(nt))
    else:
        drop |= set(range(nt) if abs(m) >= 2 else range(1, nt))
    keep = np.array([i for i in range(R.size) if i not in drop])
    if keep.size != field.size:
        raise NotImplementedError(
            f"staggered: {field.name} keeps {field.size} of {R.size} nodes; "
            f"the wall-reduction rule of this library accounts for {keep.size}")
    return R[keep], T[keep]


def _stag_values(value, field, grids, geometry, m):
    R, T = _stag_nodes(field, grids, geometry, m)
    if isinstance(value, vc.Vec):
        return evaluate(value.c[_STAG_COMPONENT[field.name[-2:]]], R, T)
    return evaluate(value, R, T)


def _stag_state(cm, sources, geometry, m):
    x = np.zeros(cm.A_total.shape[1], dtype=complex)
    for field in cm.layout.fields:
        base = field.name[:-2] if field.name[-2:] in _STAG_COMPONENT else field.name
        value = sources.get(base)
        if value is None:
            continue
        x[field.offset:field.offset + field.size] = _stag_values(
            value, field, cm.grids, geometry, m)
    return x


def _stag_solve(cm, x_f):
    """Solve the C-grid pencil and say which rows carry a solution.

    The theta-face component's theta = 0 and theta = pi rings are not
    degrees of freedom (zero face area, zero volume weight); the
    assembler pins them with a Dirichlet row and no mass, and they are
    left out of the error norm. A row left empty by the assembler is
    pinned here as well rather than making the solve singular.
    """
    A, B = cm.A_total.tocsr(), cm.B_total.tocsr()
    live = np.diff(A.indptr) > 0
    b = -(B @ x_f)
    dead = np.nonzero(~live)[0]
    if dead.size:
        pin = sps.csr_matrix((np.ones(dead.size), (dead, dead)), shape=A.shape)
        A, b[dead] = (A + pin).tocsr(), 0.0
    live[(cm.constraints or {}).get("pinned", ())] = False
    return spla.spsolve(A.tocsc(), b), live


def _run_staggered(case, geometry, m, n, order=2, aspect=2):
    pb, f_u, f_w = build_problem(case, geometry, m, "evolution")
    grid = _fd_grid(geometry, n, aspect)
    cm = pb.compile(grid, m=m, backend="fd2", discretization="staggered",
                    order_r=order, order_theta=order)
    sources = {"u": f_u} if not case.two_field else {"u": f_u, "w": f_w}
    x, live = _stag_solve(cm, _stag_state(cm, sources, geometry, m))
    u_manu = manufactured(geometry, m, case.unknown)
    err_u, err_w, scale = 0.0, 0.0, 0.0
    for field in cm.layout.fields:
        base = field.name[:-2] if field.name[-2:] in _STAG_COMPONENT else field.name
        sl = slice(field.offset, field.offset + field.size)
        got, keep = x[sl], live[sl]
        if base == "u":
            exact = _stag_values(u_manu, field, cm.grids, geometry, m)
            scale = max(scale, float(np.max(np.abs(exact))))
            err_u = max(err_u, _err(got[keep], exact[keep]))
        else:
            err_w = max(err_w, _err(got[keep], 0.0))
    return Result(err_u / scale, err_w / scale, scale, case.two_field)


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

class CoreSpec:
    """How a core is run and what its convergence is expected to be.

    ``order`` is the formal order of the scheme, or None for a spectral
    core, which is judged instead by ``tolerance`` on the finer of the
    two resolutions together with a decrease from the coarser one.
    ``min_order`` is the order a cell must actually reach at the matrix
    resolutions, below the formal one where the scheme is still
    pre-asymptotic there.
    """

    __slots__ = ("name", "runner", "resolutions", "order", "min_order",
                 "tolerance", "label")

    def __init__(self, name, runner, resolutions, order, tolerance, label,
                 min_order=None):
        self.name, self.runner, self.resolutions = name, runner, resolutions
        self.order, self.tolerance, self.label = order, tolerance, label
        self.min_order = (None if order is None
                          else order - 0.5 if min_order is None else min_order)


CORE_SPECS = {
    "collocated": CoreSpec("collocated", lambda c, g, m, n: _run_collocated(c, g, m, n, 4, 2),
                           (12, 24), 4, 1e-2, "collocated FD, order 4",
                           min_order=3.0),
    "staggered": CoreSpec("staggered", lambda c, g, m, n: _run_staggered(c, g, m, n, 2, 2),
                          (12, 24), 2, 1e-1, "staggered Arakawa-C, order 2"),
    "sh_fd": CoreSpec("sh_fd", lambda c, g, m, n: _run_sh_fd(c, g, m, n),
                      (32, 64), 4, 1e-3, "spin-weighted SH, radial FD order 4",
                      min_order=3.0),
    "dedalus": CoreSpec("dedalus", lambda c, g, m, n: _run_dedalus(c, g, m, n),
                        (8, 14), None, 1e-8, "Dedalus spectral"),
}


class Outcome:
    """One cell of the coverage matrix."""

    __slots__ = ("status", "errors", "resolutions", "order", "reason")

    def __init__(self, status, errors=(), resolutions=(), order=float("nan"), reason=""):
        self.status, self.errors = status, tuple(errors)
        self.resolutions, self.order, self.reason = tuple(resolutions), order, reason

    def __repr__(self):
        if self.status == "PASS":
            return f"Outcome(PASS, order={self.order:.2f}, errors={self.errors})"
        return f"Outcome({self.status}, {self.reason})"


def measure(case, core, geometry, m):
    """Run one cell: PASS with a measured order, REFUSED, or FAIL.

    A core that declines the operator raises NotImplementedError or
    TypeError from its own compiler; anything else — a singular
    assembled operator, a solve that does not converge — is a failure of
    the core, not a reason to skip.
    """
    spec = CORE_SPECS[core]
    errors = []
    for n in spec.resolutions:
        try:
            errors.append(spec.runner(case, geometry, m, n).error)
        except (NotImplementedError, TypeError) as exc:
            return Outcome("REFUSED", reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return Outcome("FAIL", reason=f"{type(exc).__name__}: {exc}")
    return verdict(core, errors)


def verdict(core, errors):
    """The cell's verdict from its two errors, under the core's spec."""
    spec = CORE_SPECS[core]
    if not all(np.isfinite(errors)):
        return Outcome("FAIL", errors, spec.resolutions, reason="non-finite error")
    order = order_of(errors[0], errors[1], *spec.resolutions)
    ok = (errors[1] <= spec.tolerance and errors[1] < errors[0]
          and (spec.min_order is None or order >= spec.min_order))
    return Outcome("PASS" if ok else "FAIL", errors, spec.resolutions, order,
                   "" if ok else "does not meet the expected order or tolerance")


def run_once(case, core, geometry, m):
    """One solve at the coarse resolution; the core's own errors escape."""
    spec = CORE_SPECS[core]
    return spec.runner(case, geometry, m, spec.resolutions[0])
