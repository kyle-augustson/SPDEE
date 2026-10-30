"""Translate the IR into a Dedalus v3 eigenvalue problem.

    FieldRef(unknown)                -> Field / VectorField / TensorField on the basis
    FieldRef(background)             -> NCC field on the meridional basis (shell) or
                                        grid values (ball, see below)
    Sum / Negation / ScalarMultiple  -> +, -, *
    Grad, Div, Curl, Lap             -> d3.grad, d3.div, d3.curl, d3.lap
    LapH(u)                          -> lap(u) minus its radial part (``_laph``)
    Cross, Dot, Transpose            -> d3.cross, @, d3.transpose
    Dt(X)                            -> s_eig * X

Boundary conditions become interpolated rows ``expr(r=r_b) = 0`` enforced
with tau fields: the number of boundary rows on an unknown selects its tau
layout (``_tau_plan``), lifted into the equation that owns the unknown.
Gauge unknowns get a basis-free tau in their constraint equation and
``integ(p) = 0``.  A row is resolved term by term on both geometries
(``_wall``): each unit-vector projection is a component operator of its
operand at the wall, each other term is interpolated there, a coefficient
is a number at the wall, and the rows of one expression that differ only
in theta_hat / phi_hat are one angular row.  A tensor unknown has no
radial derivative in the IR, so it lifts no tau and its rows are handed to
the vector its equation differentiates (``_tau_owner``).  A written
involution ``Div(b) = 0`` takes the ``DivergenceCleaning`` of
``spdee.ir.constraints``: an auto multiplier through + Grad in the curl
evolution, gauged by ``integ`` (``_with_divergence_cleaning``).

Ball.  A BallBasis holds only radial NCCs and Dedalus does not assemble
l-coupled ball subproblems, so the ball pencil is put together here.  Every
background is kept as grid values; a theta-independent regular one becomes
a radial-basis NCC, any other one multiplies an unknown through an auxiliary
variable ``w`` with the row ``w = 0``.  Dedalus builds the l-decoupled blocks
for the requested m, each ``w`` is then probed column by column in grid space
(exact on the polynomial space of the basis) and eliminated,
``A = R L T``, ``B = R M T``.  An identically zero background drops its
term.  Unit vectors translate to component operators in boundary rows
only: a row is a linear combination of projections and ordinary terms,
each projection ``Dot(r_hat, X)`` is ``radial(X(r=R))``, each other term
is interpolated to the wall, and a theta/phi pair of rows on one
expression is a single ``angular`` row; a coefficient is taken at the
wall.  In the bulk the projection is refused (``v_r`` of a regular vector
is not a regular scalar at the origin) except as ``f(r) * Dot(r_hat, X)``,
which becomes ``Dot(f(r) r_hat, X)`` with a radial-basis NCC.  Taus lift
onto the ball basis, a radial tau through ``r r_hat``.

Pencil: Dedalus solves ``M dt(X) + L X = 0`` with ``dt -> s``.  Returning
``K = L_min`` and ``M = M_min`` gives ``lambda = -s`` and, with spdee's
``omega = -i lambda``, the Dedalus relation ``s = -i omega``.
"""

from __future__ import annotations

import dataclasses

import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import sympy

from spdee.ir.expressions import (
    Expression, FieldRef, Sum, Negation, ScalarMultiple, AsSymmetric, _BackgroundScalarValue,
)
from spdee.ir.operators import Grad, Div, Curl, Lap, LapH, Dt, Cross, Dot, Transpose
from spdee.ir.equations import Boundary, Equation, EquationSystem
from spdee.ir.fields import Unknown, r as R_SYM, theta as TH_SYM
from spdee.ir.backgrounds import is_sympy_component, get_sympy_expr
from spdee.ir.constraints import classify
from spdee.ir.walker import walk_equation_system

__all__ = ["DedalusGrid", "DedalusCompiledProblem", "compile_dedalus"]

_COMP_INDEX = {"phi": 0, "ph": 0, "theta": 1, "th": 1, "r": 2}
_RANK = {"scalar": 0, "vector": 1, "tensor": 2}
_UNIT_NAMES = {"r_hat": "r", "theta_hat": "theta", "phi_hat": "phi"}


@dataclass
class DedalusGrid:
    """Resolution and geometry of the shell or ball basis.  ``N_phi`` defaults
    to the smallest even count that carries m; ``r_inner = 0`` selects a ball."""
    N_r: int
    N_theta: int
    r_inner: float = 0.0
    r_outer: float = 1.0
    geometry: Optional[str] = None
    N_phi: Optional[int] = None
    dealias: float = 1.0

    def __post_init__(self):
        if self.geometry is None:
            self.geometry = "ball" if self.r_inner == 0.0 else "shell"
        if self.geometry not in ("ball", "shell"):
            raise ValueError("geometry must be 'ball' or 'shell'")

    @property
    def is_ball(self) -> bool:
        return self.geometry == "ball"

    @classmethod
    def from_grid(cls, grid, **kw) -> "DedalusGrid":
        """Same N_r, N_theta and radii as an fd2 Grid."""
        r_in = 0.0 if getattr(grid, "is_ball", False) else float(grid.r[0])
        return cls(N_r=int(getattr(grid, "N_r", len(grid.r))),
                   N_theta=int(getattr(grid, "N_theta", len(grid.theta))),
                   r_inner=r_in, r_outer=float(grid.r[-1]), **kw)


def _as_dedalus_grid(grid) -> DedalusGrid:
    if isinstance(grid, DedalusGrid):
        return grid
    if isinstance(grid, dict):
        return DedalusGrid(**grid)
    if hasattr(grid, "r") and hasattr(grid, "theta"):
        return DedalusGrid.from_grid(grid)
    raise TypeError("grid must be a DedalusGrid, a dict of its fields, or an fd2 Grid")


def _tau_plan(kind: str, n_bc: int, ball: bool = False) -> List[Tuple[str, int]]:
    """Tau fields for an unknown with ``n_bc`` scalar boundary rows as
    (tau kind, lift order): 'vector' lifts a sphere VectorField, 'radial'
    lifts a sphere scalar into the radial component.  A ball has one wall,
    so a vector takes at most three rows there.  A tensor unknown has no
    radial derivative in the IR (Grad and Lap refuse rank 2), so it lifts
    nothing: its rows are handed to its vector by ``_tau_owner``."""
    if kind == "scalar":
        if n_bc <= 2:
            return [("scalar", -1 - i) for i in range(n_bc)]
    elif kind == "tensor":
        if n_bc == 0:
            return []
    elif kind == "vector":
        table = {
            0: [],
            1: [("radial", -1)],
            2: [("radial", -1), ("radial", -2)],
            3: [("vector", -1)],
            4: [("vector", -1), ("radial", -2)],
            6: [("vector", -1), ("vector", -2)],
        }
        if ball:
            table = {k: v for k, v in table.items() if k <= 3}
        if n_bc in table:
            return table[n_bc]
    supported = "scalar 0-2, vector 0-3" if ball else "scalar 0-2, vector 0,1,2,3,4,6"
    raise NotImplementedError(f"no tau layout for a {kind} unknown with {n_bc} boundary rows"
                              f"{' on a ball' if ball else ''}; supported: {supported}")


class _GridNCC:
    """A background on the ball: values on the local (1, N_theta, N_r) grid
    behind ``rank`` leading component axes in Dedalus order (phi, theta, r);
    ``unit`` names a unit vector."""

    def __init__(self, data, rank: int, unit: Optional[str] = None):
        self.data = np.asarray(data)
        self.rank = rank
        self.unit = unit


class _Component:
    """``Dot(unit vector, X)`` on the ball, resolved at the wall into
    ``radial(X(r=R))`` or, for a theta/phi pair on one expression, into a
    single ``angular(X(r=R))`` row; ``operand`` may itself be a _Component.
    In the bulk the projection is refused: the radial component of a
    regular vector is r^(l-1) Y_l near the origin (v = e_z gives
    v_r = cos(theta)), not a regular scalar, and the tangential ones are
    singular at the poles."""

    def __init__(self, comp: str, operand, index: int, key: str):
        self.comp, self.operand, self.index, self.key = comp, operand, index, key

    def refuse(self):
        why = ("the radial component of a regular vector is not a regular scalar at the "
               "origin; write f(r)*dot(r_hat, X) with f(0) = 0, or f(r)*r_hat as a vector "
               "background" if self.comp == "r" else
               f"a {self.comp} component is singular at the poles")
        raise NotImplementedError(
            f"Dot({self.comp}_hat, {self.key}): unit-vector projections are available on the "
            f"ball only in boundary rows (theta_hat and phi_hat as a pair on one expression): "
            f"{why}")

    def at_wall(self, d3, rb):
        inner = self.operand
        base = inner.at_wall(d3, rb) if isinstance(inner, _Component) else inner(r=rb)
        if self.comp == "r":
            return d3.radial(base, index=self.index)
        return d3.angular(base, index=self.index)


@dataclass
class _AuxTerm:
    field: Any
    kind: str
    ncc: np.ndarray
    rank_ncc: int
    op: Any
    rank_op: int
    ncc_first: bool
    refs: List[Any]


def _values(x) -> Tuple[Any, int]:
    if isinstance(x, _GridNCC):
        return x.data, x.rank
    return complex(x), 0


def _is_zero(x) -> bool:
    """A numeric zero factor: the term is dropped, as Dedalus's Multiply does."""
    return isinstance(x, (int, float, complex)) and x == 0


def _pointwise(kind: str, a, ra: int, b, rb: int):
    """Product of grid arrays with leading component axes.  The Dedalus
    component order (phi, theta, r) is left-handed, so the cross product is
    the negative of the cyclic formula on the indices."""
    if kind == "mul":
        return a * b
    if kind == "cross":
        return -np.stack([a[1] * b[2] - a[2] * b[1],
                          a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0]])
    letters = "ijkl"
    sa = letters[:ra]
    sb = letters[ra - 1] + letters[ra:ra + rb - 1]
    out = letters[:ra - 1] + letters[ra:ra + rb - 1]
    return np.einsum(f"{sa}...,{sb}...->{out}...", a, b)


def _row_size(row) -> int:
    """Scalar rows of a Dedalus wall row: the product of its index dimensions."""
    return int(np.prod([cs.dim for cs in row.tensorsig], dtype=int))


def _selected(P) -> np.ndarray:
    """Column of the single nonzero in each row of a selection matrix."""
    P = sp.csr_matrix(P)
    return P.indices[P.indptr[:-1]]


def _selection(rows, n_rows, n_cols=None, transpose=False):
    idx = np.asarray(rows)
    n_cols = idx.size if n_cols is None else n_cols
    ones = np.ones(idx.size)
    if transpose:
        return sp.csr_matrix((ones, (np.arange(idx.size), idx)), shape=(idx.size, n_rows))
    return sp.csr_matrix((ones, (idx, np.arange(idx.size))), shape=(n_rows, n_cols))


class DedalusCompiledProblem:
    """An EquationSystem compiled to the Dedalus pencil of one m.

    ``A_total`` is ``L_min``, ``B_total`` is ``M_min`` (block-diagonal over
    the l-keyed Dedalus subproblems of m when they decouple); ``fields`` maps
    names to Dedalus fields; ``assemble_evp_matrices`` and ``solve_evp``
    mirror the fd2 interface.
    """

    discretization = "dedalus"

    def __init__(self, system: EquationSystem, grid, m: int, params: Optional[dict] = None,
                 ncc_cutoff: float = 1e-8, comm=None, dtype=np.complex128):
        try:
            import dedalus.public as d3
        except ImportError as exc:
            raise ImportError("the dedalus backend needs Dedalus v3 (the 'dedalus' tier of "
                              "docs/requirements.md; scripts/install_dedalus_env.sh builds it); "
                              f"the fd2 and sh_fd backends run without it: {exc}") from exc
        self.d3 = d3
        self.grid = _as_dedalus_grid(grid)
        self.m = int(m)
        self.params = {sympy.sympify(k): v for k, v in (params or {}).items()}
        self.dtype = dtype
        self.multipliers: Dict[str, str] = {}
        self.system = self._with_divergence_cleaning(system)
        system = self.system
        g = self.grid

        N_phi = g.N_phi or max(8, 2 * abs(self.m) + 4)
        N_phi += N_phi % 2
        coords = d3.SphericalCoordinates("phi", "theta", "r")
        dist = d3.Distributor(coords, dtype=dtype, comm=comm)
        if g.is_ball:
            basis = d3.BallBasis(coords, shape=(N_phi, g.N_theta, g.N_r), radius=g.r_outer,
                                 dealias=g.dealias, dtype=dtype)
            sphere = basis.surface
            self.bm = basis.radial_basis
            lift_basis = basis
        else:
            basis = d3.ShellBasis(coords, shape=(N_phi, g.N_theta, g.N_r), radii=(g.r_inner, g.r_outer),
                                  dealias=g.dealias, dtype=dtype)
            sphere = basis.outer_surface
            self.bm = basis.meridional_basis
            lift_basis = basis.derivative_basis(1)
        self.coords, self.dist, self.basis, self.sphere = coords, dist, basis, sphere
        phi, theta, r = dist.local_grids(basis)
        self._R, self._TH = np.broadcast_arrays(r, theta)
        self.r_grid, self.theta_grid = r.ravel(), theta.ravel()
        self._lift = lambda A, n: d3.Lift(A, lift_basis, n)

        self.fields: Dict[str, Any] = {}
        self._ncc_cache: Dict[Any, Any] = {}
        self._aux: List[_AuxTerm] = []
        self._aux_cache: Dict[Any, Any] = {}
        self.s_eig = dist.Field(name="s_eig")
        self.unknown_names: List[str] = []
        for name, f in system.unknowns.items():
            self.fields[name] = self._new_field(_RANK[f.kind], name)
            self.unknown_names.append(name)
        for name, f in system.backgrounds.items():
            self.fields[name] = self._make_background(f)
        if g.is_ball:
            self._er = self._new_field(1, "r_vec", bases=self.bm)
            self._er["g"][2] = r
        else:
            self._er = self._unit_vector("r")

        eq_lhs = [self._weighted(eq) for eq in system.equations]

        self.tau_fields: List[Any] = []
        bc_rows = self._boundary_rows()
        n_bc: Dict[str, int] = {n: 0 for n in self.unknown_names}
        for _expr, owner, n_rows in bc_rows:
            n_bc[owner] += n_rows
        for name in self.unknown_names:
            plan = _tau_plan(system.unknowns[name].kind, n_bc[name], ball=g.is_ball)
            if not plan:
                continue
            host = self._host_equation(name)
            for j, (tkind, order) in enumerate(plan):
                tau, term = self._tau_term(tkind, order, f"tau_{name}{j + 1}")
                self.tau_fields.append(tau)
                eq_lhs[host] = eq_lhs[host] + term
        gauge_rows = []
        gauge_hosts: List[int] = []
        for gname in system.gauge_unknowns:
            host = self._constraint_equation(gname, gauge_hosts)
            gauge_hosts.append(host)
            tau = dist.Field(name=f"tau_{gname}_gauge")
            self.tau_fields.append(tau)
            eq_lhs[host] = eq_lhs[host] + tau
            gauge_rows.append(d3.integ(self.fields[gname]))

        variables = ([self.fields[n] for n in self.unknown_names] + [t.field for t in self._aux]
                     + self.tau_fields)
        problem = d3.EVP(variables, eigenvalue=self.s_eig, namespace={})
        for lhs in eq_lhs:
            problem.add_equation((lhs, 0))
        for expr, _owner, _n in bc_rows:
            problem.add_equation((expr, 0))
        for row in gauge_rows:
            problem.add_equation((row, 0))
        n_eq = len(eq_lhs) + len(bc_rows) + len(gauge_rows)
        for t in self._aux:
            problem.add_equation((t.field, 0))
        self._aux_eqs = list(range(n_eq, n_eq + len(self._aux)))
        self.problem = problem
        self.solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
        self._assemble()

    # ---- constraints, fields and backgrounds

    def _with_divergence_cleaning(self, system: EquationSystem) -> EquationSystem:
        """The system with the ``DivergenceCleaning`` of every written
        involution (``spdee.ir.constraints``): the multiplier enters the
        curl evolution through + Grad and is gauged by ``integ``, which
        Dedalus keeps in the l = 0 block alone, so the record's m = 0 gauge
        needs no switch.  The record's wall pins on the multiplier are not
        taken: on a tau core they would be rows beyond the second-order
        count of the evolution, and the gauge fixes the multiplier (zero on
        the exact solution).  A system outside the classifier's vocabulary,
        including a hand-written multiplier beside a curl evolution, is
        compiled as written."""
        try:
            record = classify(system, walk_equation_system(system, self.m),
                              geometry=self.grid.geometry, mimetic=False, core="dedalus")
        except NotImplementedError:
            return system
        cleaning = [dc for dc in record.divergence_cleaning.values() if dc.constraint is not None]
        if not cleaning:
            return system
        equations = list(system.equations)
        gauge = list(system.gauge_unknowns)
        for dc in cleaning:
            i = next(i for i, eq in enumerate(equations) if eq.name == dc.evolution)
            lhs = equations[i].lhs + Grad(FieldRef(Unknown.Scalar(dc.multiplier)))
            equations[i] = dataclasses.replace(equations[i], lhs=lhs)
            gauge.append(dc.multiplier)
            self.multipliers[dc.vector] = dc.multiplier
        return EquationSystem(equations, list(system.boundary_conditions), gauge_unknowns=gauge)

    def _new_field(self, rank: int, name: str, bases=None):
        dist, coords = self.dist, self.coords
        bases = self.basis if bases is None else bases
        if rank == 0:
            return dist.Field(name=name, bases=bases)
        if rank == 1:
            return dist.VectorField(coords, name=name, bases=bases)
        return dist.TensorField((coords, coords), name=name, bases=bases)

    def _weighted(self, eq):
        """The Dedalus operand of an equation, multiplied by its radial
        ``weight`` when it has one; the taus are added after.  The weight is
        folded into the background coefficient of each term (`_fold`), so
        that w(r) dlnrho(r) dot(er, v) is one NCC of grid values w dlnrho,
        not w(r) times the NCC product dlnrho dot(er, v): a weight chosen to
        make the coefficients low-degree polynomials (r^2 H_rho dlnrho =
        -r^2) then leaves no steep factor for ``ncc_cutoff`` to truncate."""
        if eq.weight is None:
            return self._operand(eq.lhs)
        if self.grid.is_ball:
            raise NotImplementedError(f"equation {eq.name!r}: equation weights are implemented on the shell only")
        return self._fold(self._eval_sympy(eq.weight), eq.lhs)

    def _fold(self, w: np.ndarray, e: Expression):
        """``w * e`` with the grid values ``w`` multiplied into the scalar
        background coefficients of every term of ``e``: through sums,
        negations and Dt, along scalar coefficients, and onto the unknown
        side of a product with a background vector (w g S er as (w g) S er).
        A dot or cross product with a background stays whole behind its
        merged scalar ((2 w) cross(ez, v), w dot(U0, grad(v))), as in the
        reference implementation, as does a term with no background
        factor."""
        if isinstance(e, Sum):
            return self._fold(w, e.a) + self._fold(w, e.b)
        if isinstance(e, Negation):
            return -self._fold(w, e.operand)
        if isinstance(e, AsSymmetric):
            return self._fold(w, e.operand)
        if isinstance(e, Dt):
            return self.s_eig * self._fold(w, e.operand)
        if isinstance(e, ScalarMultiple):
            s, op = e.scalar, e.operand
            if s.is_background and not op.is_background:
                return self._fold(w * self._bg_values(s), op)
            if op.is_background and not s.is_background:
                if op.is_scalar:
                    return self._fold(w * self._bg_values(op), s)
                return self._dedalus_product("mul", self._fold(w, s), self._translate(op))
        return self._dedalus_product("mul", self._merged_ncc(w), self._operand(e))

    def _bg_values(self, e: Expression):
        """Grid values of a background expression on the shell: a number or
        an array (components first, then the meridional grid)."""
        x = self._translate(e)
        if isinstance(x, (int, float, complex)):
            return x
        if not isinstance(x, self.d3.Field):
            x = x.evaluate()
        x.change_scales(1)
        return np.array(x["g"])

    def _merged_ncc(self, w):
        """The scalar NCC field of grid values ``w``; 0 when they vanish."""
        data = np.broadcast_to(np.asarray(w, dtype=self.dtype), self._R.shape)
        if not np.any(data):
            return 0
        key = ("__merged", data.tobytes())
        if key not in self._ncc_cache:
            fld = self._new_field(0, f"ncc{len(self._ncc_cache)}", bases=self.bm)
            fld["g"] = data
            self._ncc_cache[key] = fld
        return self._ncc_cache[key]

    def _tau_term(self, tkind: str, order: int, name: str):
        """A tau field on the sphere and its lifted term: 'scalar' and
        'vector' lift a field of that rank, 'radial' a scalar into the
        radial component (through r r_hat on a ball)."""
        tau = self._new_field(1 if tkind == "vector" else 0, name, bases=self.sphere)
        lifted = self._lift(tau, order)
        return tau, self._er * lifted if tkind == "radial" else lifted

    def _eval_component(self, comp) -> np.ndarray:
        """A BackgroundComponent on the meridional grid, shape (1, N_theta, N_r)."""
        if is_sympy_component(comp):
            return self._eval_sympy(get_sympy_expr(comp))
        return np.asarray(comp.evaluate_on_grid(self._R, self._TH), dtype=self.dtype)

    def _eval_sympy(self, expr) -> np.ndarray:
        expr = sympy.sympify(expr).subs(self.params)
        free = expr.free_symbols - {R_SYM, TH_SYM}
        if free:
            raise ValueError(f"background expression {expr} has unbound symbols {free}; pass them in params")
        fn = sympy.lambdify((R_SYM, TH_SYM), expr, modules="numpy")
        val = fn(self._R, self._TH)
        return np.broadcast_to(np.asarray(val, dtype=self.dtype), self._R.shape).copy()

    def _background_values(self, f) -> np.ndarray:
        """Component values on the meridional grid behind the component axes."""
        if f.kind == "scalar":
            return self._eval_component(f.component)
        if f.kind == "vector":
            out = np.zeros((3,) + self._R.shape, dtype=self.dtype)
            for label, comp in f.components.items():
                out[_COMP_INDEX[label.split("_", 1)[1]]] = self._eval_component(comp)
            return out
        out = np.zeros((3, 3) + self._R.shape, dtype=self.dtype)
        for key, comp in f.components.items():
            i = _COMP_INDEX[str(key[0]).lower()]
            j = _COMP_INDEX[str(key[1]).lower()]
            val = self._eval_component(comp) if hasattr(comp, "evaluate_on_grid") else self._eval_sympy(comp)
            out[i, j] = val
            if getattr(f, "symmetric", False) and i != j:
                out[j, i] = val
        return out

    def _make_background(self, f):
        data = self._background_values(f)
        if self.grid.is_ball:
            unit = _UNIT_NAMES.get(f.name)
            if unit is not None:
                expected = np.zeros(3)
                expected[_COMP_INDEX[unit]] = 1.0
                if data.shape[0] != 3 or np.any(data.reshape(3, -1) != expected[:, None]):
                    unit = None
            return _GridNCC(data, _RANK[f.kind], unit)
        fld = self._new_field(_RANK[f.kind], f.name, bases=self.bm)
        fld["g"] = data
        return fld

    def _unit_vector(self, comp: str):
        key = f"__unit_{comp}"
        if key not in self._ncc_cache:
            fld = self.dist.VectorField(self.coords, name=f"e_{comp}", bases=self.bm)
            fld["g"] = 0.0
            fld["g"][_COMP_INDEX[comp]] = 1.0
            self._ncc_cache[key] = fld
        return self._ncc_cache[key]

    def _position_vector(self):
        """The NCC x = r r_hat, regular on both geometries."""
        if self.grid.is_ball:
            return self._er
        key = "__position"
        if key not in self._ncc_cache:
            fld = self.dist.VectorField(self.coords, name="x_vec", bases=self.bm)
            fld["g"] = 0.0
            fld["g"][2] = self._R
            self._ncc_cache[key] = fld
        return self._ncc_cache[key]

    def _scalar_coefficient(self, value):
        """A SymPy scalar coefficient as a number, an NCC field (shell) or
        grid values (ball)."""
        expr = sympy.sympify(value).subs(self.params)
        if not (expr.free_symbols & {R_SYM, TH_SYM}):
            if expr.free_symbols:
                raise ValueError(f"coefficient {expr} has unbound symbols; pass them in params")
            return complex(expr)
        if self.grid.is_ball:
            return _GridNCC(self._eval_sympy(expr), 0)
        key = sympy.srepr(expr)
        if key not in self._ncc_cache:
            fld = self.dist.Field(name=f"ncc{len(self._ncc_cache)}", bases=self.bm)
            fld["g"] = self._eval_sympy(expr)
            self._ncc_cache[key] = fld
        return self._ncc_cache[key]

    def _radial_ncc(self, ncc: _GridNCC):
        """The Dedalus radial-basis field of a theta-independent background
        whose components are regular on the ball (scalar; f(r) r_hat;
        diagonal tensor with equal theta and phi entries), else None; on
        the shell the same profile on the meridional basis."""
        data = ncc.data
        scale = np.abs(data).max()
        if scale == 0:
            return 0.0
        if np.abs(data - data[..., :1, :]).max() > 1e-12 * scale:
            return None
        prof = data[..., :1, :]
        if ncc.rank == 1 and np.any(prof[:2] != 0):
            return None
        if ncc.rank == 2 and (np.any(prof[~np.eye(3, dtype=bool)] != 0) or np.any(prof[0, 0] != prof[1, 1])):
            return None
        key = (ncc.rank, prof.tobytes())
        if key not in self._ncc_cache:
            fld = self._new_field(ncc.rank, f"ncc{len(self._ncc_cache)}", bases=self.bm)
            fld["g"] = prof
            c = np.abs(fld["c"])
            c = c.reshape(-1, c.shape[-1]).max(axis=0)
            if c.max() > 0 and c[-2:].max() > 1e-6 * c.max():
                tail = c[-2:].max() / c.max()
                warnings.warn(f"a radial background is not spectrally resolved on the "
                              f"{self.grid.geometry} (coefficient tail {tail:.1e} of its peak); "
                              f"it must be smooth and regular at the origin", RuntimeWarning)
            self._ncc_cache[key] = fld
        return self._ncc_cache[key]

    # ---- translation

    def _translate(self, e: Expression):
        d3 = self.d3
        if isinstance(e, FieldRef):
            return self.fields[e.field.name]
        if isinstance(e, _BackgroundScalarValue):
            return self._scalar_coefficient(e.value)
        if isinstance(e, Sum):
            a, b = self._bulk(e.a), self._bulk(e.b)
            if isinstance(a, _GridNCC) or isinstance(b, _GridNCC):
                (ga, ra), (gb, rb) = _values(a), _values(b)
                return _GridNCC(ga + gb, max(ra, rb))
            return a + b
        if isinstance(e, Negation):
            x = self._bulk(e.operand)
            return _GridNCC(-x.data, x.rank, x.unit) if isinstance(x, _GridNCC) else -x
        if isinstance(e, ScalarMultiple):
            return self._product("mul", e.scalar, e.operand)
        if isinstance(e, AsSymmetric):
            return self._translate(e.operand)
        if isinstance(e, Dt):
            return self.s_eig * self._operand(e.operand)
        if isinstance(e, Grad):
            return d3.grad(self._operand(e.operand))
        if isinstance(e, Div):
            return d3.div(self._operand(e.operand))
        if isinstance(e, Curl):
            return d3.curl(self._operand(e.operand))
        if isinstance(e, Lap):
            return d3.lap(self._operand(e.operand))
        if isinstance(e, LapH):
            return self._laph(e)
        if isinstance(e, Cross):
            return self._product("cross", e.a, e.b)
        if isinstance(e, Dot):
            return self._product("dot", e.a, e.b)
        if isinstance(e, Transpose):
            x = self._bulk(e.operand)
            if isinstance(x, _GridNCC):
                return _GridNCC(np.swapaxes(x.data, 0, 1), x.rank)
            return d3.transpose(x)
        raise NotImplementedError(f"cannot translate IR node {type(e).__name__} to Dedalus")

    def _bulk(self, e: Expression):
        """A translated expression outside a boundary row: a unit-vector
        projection on the ball is refused wherever it is combined."""
        x = self._translate(e)
        if isinstance(x, _Component):
            x.refuse()
        return x

    def _operand(self, e: Expression):
        """A translated expression that must carry unknowns."""
        x = self._bulk(e)
        if isinstance(x, _GridNCC):
            raise NotImplementedError(f"{e!r}: a bare background cannot be differentiated or stand "
                                      f"alone on the Dedalus ball; only products with unknowns are supported")
        return x

    def _dedalus_product(self, kind: str, a, b):
        if _is_zero(a) or _is_zero(b):
            return 0
        if kind == "mul":
            return a * b
        if kind == "cross":
            return self.d3.cross(a, b)
        return a @ b

    def _product(self, kind: str, a_ir: Expression, b_ir: Expression):
        a_bg, b_bg = a_ir.is_background, b_ir.is_background
        if a_bg and b_bg:
            a, b = self._translate(a_ir), self._translate(b_ir)
            if isinstance(a, _GridNCC) or isinstance(b, _GridNCC):
                (ga, ra), (gb, rb) = _values(a), _values(b)
                rank = {"mul": max(ra, rb), "cross": 1, "dot": ra + rb - 2}[kind]
                return _GridNCC(_pointwise(kind, ga, ra, gb, rb), rank)
            return self._dedalus_product(kind, a, b)
        if not (a_bg or b_bg) or not self.grid.is_ball:
            return self._dedalus_product(kind, self._bulk(a_ir), self._bulk(b_ir))
        ncc_ir, op_ir, ncc_first = (a_ir, b_ir, True) if a_bg else (b_ir, a_ir, False)
        ncc = self._translate(ncc_ir)
        if kind == "mul" and isinstance(ncc, _GridNCC) and ncc.rank == 0 and isinstance(op_ir, Dot):
            folded = self._fold_unit(ncc_ir, op_ir)
            if folded is not None:
                return folded
        if isinstance(ncc, _GridNCC) and ncc.unit is not None:
            if kind == "dot":
                X = self._translate(op_ir)
                if isinstance(X, _GridNCC):
                    raise NotImplementedError(f"{op_ir!r}: a bare background cannot stand alone on the ball")
                return _Component(ncc.unit, X, 0 if ncc_first else _RANK[op_ir.kind] - 1, repr(op_ir))
            raise NotImplementedError(f"{ncc.unit}_hat multiplies {op_ir!r} on the ball: unit vectors are "
                                      f"singular at the origin; write f(r)*r_hat as a vector background "
                                      f"with v_r = f(r), f(0) = 0")
        if isinstance(ncc, _GridNCC):
            radial = self._radial_ncc(ncc)
            if radial is None:
                return self._aux_product(kind, ncc, op_ir, ncc_first)
            ncc = radial
        X = self._operand(op_ir)
        return self._dedalus_product(kind, ncc, X) if ncc_first else self._dedalus_product(kind, X, ncc)

    def _fold_unit(self, f_ir: Expression, dot_ir: Dot):
        """``f(r) * Dot(r_hat, X)`` on the ball as ``Dot(f(r) r_hat, X)``: the
        product is regular when f(0) = 0 although the bare projection is
        not, and f(r) r_hat is a radial-basis NCC; None when ``dot_ir``
        holds no r_hat."""
        for unit_ir, op_ir in ((dot_ir.a, dot_ir.b), (dot_ir.b, dot_ir.a)):
            if self._unit_name(unit_ir) == "r":
                scaled = ScalarMultiple(f_ir, unit_ir)
                return self._product("dot", scaled, op_ir) if unit_ir is dot_ir.a \
                    else self._product("dot", op_ir, scaled)
        return None

    def _aux_product(self, kind: str, ncc: _GridNCC, op_ir: Expression, ncc_first: bool):
        """Stand-in variable for a theta-dependent background times an
        expression of unknowns; ``Dt`` may only wrap the whole operand."""
        dt = isinstance(op_ir, Dt)
        if dt:
            op_ir = op_ir.operand
        if self._has_any_dt(op_ir):
            raise NotImplementedError(f"{op_ir!r}: Dt must be the outermost operation of a term multiplied "
                                      f"by a theta-dependent background on the ball")
        key = (kind, ncc.data.tobytes(), ncc.rank, repr(op_ir), ncc_first)
        if key not in self._aux_cache:
            n0 = len(self._aux)
            X = self._operand(op_ir)
            self._aux_cache[key] = self._aux_field(kind, ncc, X, _RANK[op_ir.kind],
                                                   self._aux_refs(op_ir, n0), ncc_first)
        w = self._aux_cache[key]
        return self.s_eig * w if dt else w

    def _aux_field(self, kind: str, ncc: _GridNCC, X, rank_op: int, refs: List[Any],
                   ncc_first: bool):
        """The auxiliary variable standing for the grid-space product of
        ``ncc`` with the translated operand ``X``."""
        rank = {"mul": max(ncc.rank, rank_op), "cross": 1, "dot": ncc.rank + rank_op - 2}[kind]
        w = self._new_field(rank, f"aux{len(self._aux)}")
        self._aux.append(_AuxTerm(w, kind, ncc.data, ncc.rank, X, rank_op, ncc_first, refs))
        return w

    def _aux_refs(self, e: Expression, n0: int) -> List[Any]:
        """The pencil columns an auxiliary term depends on: the unknowns of
        ``e`` and the auxiliaries created while translating it."""
        return [self.fields[n] for n in self._referenced_unknowns(e)] + \
            [t.field for t in self._aux[n0:]]

    def _laph(self, e: LapH):
        """``LapH(u) = lap(u) - (1/r^2) d_r(r^2 d_r u)`` with the radial part
        written through x = r r_hat, ``r^2 (u_rr + 2 u_r / r) = 2 x.grad(u)
        + x.grad(grad(u)).x``, regular on the ball.  The factor 1/r^2 is
        an NCC on the shell; on the ball it is probed in grid space, the
        operator being singular at the origin on the regular fields
        (``LapH(r^l Y_l) = -l(l+1) r^(l-2) Y_l``, so ``LapH(z) = -2 z / r^2``),
        which makes the ball LapH the pointwise operator at the quadrature
        nodes."""
        d3 = self.d3
        n0 = len(self._aux)
        u = self._operand(e.operand)
        x = self._position_vector()
        radial = 2 * (x @ d3.grad(u)) + x @ (d3.grad(d3.grad(u)) @ x)
        if not self.grid.is_ball:
            return d3.lap(u) - self._scalar_coefficient(1 / R_SYM ** 2) * radial
        if self._has_any_dt(e.operand):
            raise NotImplementedError(f"{e!r}: Dt must be the outermost operation of a LapH term "
                                      f"on the ball")
        key = ("laph", repr(e.operand))
        if key not in self._aux_cache:
            ncc = _GridNCC(1.0 / self._R ** 2, 0)
            self._aux_cache[key] = self._aux_field("mul", ncc, radial, 0,
                                                   self._aux_refs(e.operand, n0), True)
        return d3.lap(u) - self._aux_cache[key]

    def _referenced_unknowns(self, e: Expression) -> List[str]:
        out: List[str] = []

        def rec(x):
            if isinstance(x, FieldRef) and x.field.is_unknown and x.field.name not in out:
                out.append(x.field.name)
            for c in x.children:
                rec(c)
        rec(e)
        return out

    # ---- boundary rows and tau ownership

    def _unit_name(self, e) -> Optional[str]:
        """The component name of a unit-vector background referenced by ``e``."""
        if not isinstance(e, FieldRef):
            return None
        f = e.field
        unit = _UNIT_NAMES.get(getattr(f, "name", None))
        if unit is None or f.kind != "vector" or f.name not in self.system.backgrounds:
            return None
        expected = np.zeros(3)
        expected[_COMP_INDEX[unit]] = 1.0
        data = self._background_values(f)
        return None if np.any(data.reshape(3, -1) != expected[:, None]) else unit

    def _wall_projection(self, e):
        """``Dot(unit vector, X)`` (or ``Dot(X, unit vector)``) in a boundary
        row as a _Component, so that the wall row is ``radial``/``angular`` of
        ``X`` rather than a contraction with a unit-vector NCC; None otherwise."""
        if not isinstance(e, Dot):
            return None
        for unit_ir, op_ir, first in ((e.a, e.b, True), (e.b, e.a, False)):
            unit = self._unit_name(unit_ir)
            if unit is not None:
                inner = self._wall_projection(op_ir)
                if inner is None:
                    inner = self._translate(op_ir)
                if isinstance(inner, _GridNCC):
                    raise NotImplementedError(f"{op_ir!r}: a bare background cannot stand alone "
                                              f"on the ball")
                return _Component(unit, inner, 0 if first else _RANK[op_ir.kind] - 1, repr(op_ir))
        return None

    def _wall(self, e: Expression, rb: float):
        """A boundary expression as (Dedalus row at ``r = rb``, signature):
        every unit-vector projection is a component operator of its
        operand at the wall, every other term is interpolated there, a
        coefficient is taken at the wall.  The signature is the tuple of
        tangential units along the projection chain, outermost first,
        empty for a scalar row and None for a dropped term; the terms of
        one row must share it."""
        if isinstance(e, Sum):
            a, sa = self._wall(e.a, rb)
            b, sb = self._wall(e.b, rb)
            if sa is not None and sb is not None and sa != sb:
                raise NotImplementedError(f"{e!r}: a tangential projection shares a boundary row "
                                          f"only with projections on the same unit vector")
            if sa is None:
                return b, sb
            if sb is None:
                return a, sa
            return a + b, sa
        if isinstance(e, Negation):
            x, sig = self._wall(e.operand, rb)
            return -x, sig
        if isinstance(e, AsSymmetric):
            return self._wall(e.operand, rb)
        if isinstance(e, ScalarMultiple) and e.scalar.is_background:
            c = self._wall_coefficient(e.scalar, rb)
            x, sig = self._wall(e.operand, rb)
            if _is_zero(c) or sig is None:
                return 0, None
            return c * x, sig
        x = self._wall_projection(e)
        return self._row_term(self._translate(e) if x is None else x, e, rb)

    def _row_term(self, x, e: Expression, rb: float):
        """A translated term at the wall with its signature."""
        if isinstance(x, _Component):
            sig, inner = [], x
            while isinstance(inner, _Component):
                if inner.comp != "r":
                    sig.append(inner.comp)
                inner = inner.operand
            return x.at_wall(self.d3, rb), tuple(sig)
        if isinstance(x, _GridNCC):
            raise ValueError(f"{e!r}: a bare background in a boundary row")
        if _is_zero(x):
            return 0, None
        return x(r=rb), ()

    def _wall_coefficient(self, e: Expression, rb: float):
        """A background scalar coefficient of a boundary row as a number: a
        SymPy expression evaluated at the wall, any other radial profile
        interpolated there."""
        x = self._translate(e)
        if isinstance(x, (int, float, complex)):
            return x
        if isinstance(e, _BackgroundScalarValue):
            expr = sympy.sympify(e.value).subs(self.params)
            if TH_SYM not in expr.free_symbols:
                return complex(expr.subs(R_SYM, rb))
        if not isinstance(x, _GridNCC):
            x = x.evaluate() if not isinstance(x, self.d3.Field) else x
            x = _GridNCC(x["g"], 0)
        radial = self._radial_ncc(x)
        if radial is None:
            raise NotImplementedError(f"{e!r}: a theta-dependent coefficient in a boundary row "
                                      f"couples the harmonics of the wall; only a radial "
                                      f"profile or a number is supported")
        if _is_zero(radial):
            return 0
        return complex(radial(r=rb).evaluate()["g"].ravel()[0])

    def _row_key(self, e: Expression) -> str:
        """The structure of a boundary expression with theta_hat and phi_hat
        made alike, the pairing key of the tangential rows."""
        if self._unit_name(e) in ("theta", "phi"):
            return "tangent_hat"
        if not e.children:
            return repr(e)
        return f"{type(e).__name__}({', '.join(self._row_key(c) for c in e.children)})"

    def _boundary_rows(self):
        """(Dedalus row at the wall, owning unknown, scalar row count) per
        boundary condition; the rows of one expression that differ only in
        theta_hat / phi_hat, all 2^k combinations of its k tangential
        projections, are one angular row."""
        rows = []
        pending: Dict[Any, Dict[tuple, Any]] = {}
        rank = {"scalar": 0, "vector": 1, "tensor": 2}
        for bc in self.system.boundary_conditions:
            if bc.location in (Boundary.THETA_NORTH, Boundary.THETA_SOUTH):
                raise NotImplementedError("theta boundary conditions do not apply on the Dedalus "
                                          "backend (pole regularity is built in)")
            if bc.location == Boundary.R_INNER:
                if self.grid.is_ball:
                    raise ValueError(f"BC {bc.name!r} is set at r=0, but a ball has no inner boundary: "
                                     f"regularity at the origin is built into the basis; keep only "
                                     f"the conditions at the outer wall")
                rb = self.grid.r_inner
            else:
                rb = self.grid.r_outer
            names = self._referenced_unknowns(bc.expr)
            if not names:
                raise ValueError(f"BC {bc.name!r} references no unknown")
            owner = self._tau_owner(max(names, key=lambda n: rank[self.system.unknowns[n].kind]))
            row, sig = self._wall(bc.expr, rb)
            if sig is None:
                raise ValueError(f"BC {bc.name!r} vanishes identically")
            entry = (row, owner, bc.name)
            if sig:
                key = (bc.location, self._row_key(bc.expr))
                pending.setdefault(key, {})[sig] = entry
            else:
                rows.append(entry)
        for comps in pending.values():
            k = len(next(iter(comps)))
            if set(comps) != set(itertools.product(("theta", "phi"), repeat=k)):
                name = next(iter(comps.values()))[2]
                raise NotImplementedError(f"BC {name!r}: a single theta_hat or phi_hat projection is not "
                                          f"spectral on the sphere; give both projections of the same "
                                          f"expression at the same wall")
            rows.append(comps[("theta",) * k])
        return [(row, owner, _row_size(row)) for row, owner, _name in rows]

    def _tau_owner(self, name: str) -> str:
        """A scalar multiplier without a scalar equation of its own (e.g. the
        electric potential, present only through Grad(psi) in a vector
        equation) hands its boundary rows to that equation's vector unknown;
        a tensor unknown, algebraic or relaxed in its equation, to the vector
        that equation differentiates."""
        kind = self.system.unknowns[name].kind
        if kind == "tensor":
            for eq in self.system.equations:
                if eq.kind == "tensor" and name in self._referenced_unknowns(eq.lhs):
                    vectors = [n for n in self._referenced_unknowns(eq.lhs)
                               if self.system.unknowns[n].kind == "vector"]
                    if len(vectors) == 1:
                        return vectors[0]
            raise NotImplementedError(
                f"boundary rows on the tensor unknown {name!r}: a tensor has no radial derivative "
                f"in its equation, so its rows belong to the one vector unknown its tensor "
                f"equation is written in terms of")
        if kind != "scalar":
            return name
        for eq in self.system.equations:
            if eq.kind == "scalar" and name in self._referenced_unknowns(eq.lhs):
                return name
        for eq in self.system.equations:
            if eq.kind == "vector" and name in self._referenced_unknowns(eq.lhs):
                for n in self._referenced_unknowns(eq.lhs):
                    if self.system.unknowns[n].kind == "vector" and self._has_dt_of(eq.lhs, n):
                        return n
        return name

    def _has_dt_of(self, e: Expression, name: str) -> bool:
        if isinstance(e, Dt):
            return name in self._referenced_unknowns(e.operand)
        return any(self._has_dt_of(c, name) for c in e.children)

    def _has_any_dt(self, e: Expression) -> bool:
        return isinstance(e, Dt) or any(self._has_any_dt(c) for c in e.children)

    def _host_equation(self, name: str) -> int:
        eqs = self.system.equations
        kind = self.system.unknowns[name].kind
        for i, eq in enumerate(eqs):
            if eq.kind == kind and self._has_dt_of(eq.lhs, name):
                return i
        for i, eq in enumerate(eqs):
            if eq.kind == kind and name in self._referenced_unknowns(eq.lhs):
                return i
        raise ValueError(f"no equation of kind {kind!r} references unknown {name!r}")

    def _constraint_equation(self, gname: str, taken: List[int] = ()) -> int:
        """The scalar Dt-free equation hosting the gauge tau of ``gname``:
        one not yet hosting a gauge and not referencing ``gname``."""
        eqs = self.system.equations
        for i, eq in enumerate(eqs):
            if eq.kind == "scalar" and not self._has_any_dt(eq.lhs) and i not in taken \
                    and gname not in self._referenced_unknowns(eq.lhs):
                return i
        for i, eq in enumerate(eqs):
            if eq.kind == "scalar" and not self._has_any_dt(eq.lhs):
                return i
        raise ValueError(f"gauge unknown {gname!r}: no scalar constraint equation found")

    # ---- pencil assembly

    def _assemble(self) -> None:
        """Block-diagonal pencil of the subproblems carrying m, with the
        auxiliary columns of the ball probed and eliminated."""
        solver = self.solver
        blocks = [(ss.group[1], q, ss) for q in solver.subproblems for ss in q.subsystems
                  if ss.group[0] == self.m]
        if not blocks:
            raise RuntimeError(f"no Dedalus subproblem for m={self.m}; available: "
                               f"{sorted(set(ss.group[0] for q in solver.subproblems for ss in q.subsystems))}")
        blocks.sort(key=lambda b: -1 if b[0] is None else b[0])
        self._blocks = [(q, ss) for _, q, ss in blocks]
        solver.build_matrices([q for q, _ in self._blocks], ["M", "L"])
        self._T = None
        if len(self._blocks) == 1 and not self._aux:
            q = self._blocks[0][0]
            self._offsets = [0]
            self.A_total = sp.csr_matrix(q.L_min)
            self.B_total = sp.csr_matrix(q.M_min)
            return
        vars_ = solver.problem.LHS_variables
        eqns = solver.problem.equations
        Ls, Ms, col_var, row_eq, self._offsets = [], [], [], [], []
        off = 0
        for q, _ in self._blocks:
            vs = np.cumsum([0] + [q.field_size(v) for v in vars_])
            es = np.cumsum([0] + [q.field_size(e["eqn"]) for e in eqns])
            col_var.append(np.searchsorted(vs, _selected(q.pre_right_pinv), side="right") - 1)
            row_eq.append(np.searchsorted(es, _selected(q.pre_left), side="right") - 1)
            Ls.append(q.L_min)
            Ms.append(q.M_min)
            self._offsets.append(off)
            off += q.L_min.shape[0]
        L = sp.block_diag(Ls, format="csr")
        M = sp.block_diag(Ms, format="csr")
        if not self._aux:
            self.A_total, self.B_total = L, M
            return
        col_var = np.concatenate(col_var)
        row_eq = np.concatenate(row_eq)
        N = L.shape[0]

        def index(f):
            return next(i for i, v in enumerate(vars_) if v is f)

        aux_cols = np.nonzero(np.isin(col_var, [index(t.field) for t in self._aux]))[0]
        keep = np.setdiff1d(np.arange(N), aux_cols)
        T = _selection(keep, N)
        for t in self._aux:
            wcols = np.nonzero(col_var == index(t.field))[0]
            refcols = np.nonzero(np.isin(col_var, [index(f) for f in t.refs]))[0]
            Z = self._probe(t, wcols, refcols, index(t.field))
            T = T + _selection(wcols, N) @ (sp.csr_matrix(Z) @ T[refcols, :])
        keep_rows = np.nonzero(~np.isin(row_eq, self._aux_eqs))[0]
        R = _selection(keep_rows, N, transpose=True)
        self._T = T.tocsr()
        self.A_total = (R @ L @ self._T).tocsr()
        self.B_total = (R @ M @ self._T).tocsr()

    def _probe(self, t: _AuxTerm, wcols: np.ndarray, refcols: np.ndarray, iw: int) -> np.ndarray:
        """Columns of the auxiliary term over the pencil columns it references:
        unit vector in, grid-space product, forward transform, gather."""
        solver, vars_ = self.solver, self.solver.problem.LHS_variables
        state = solver.state
        offsets = np.asarray(self._offsets)
        wpos = np.full(int(sum(q.L_min.shape[0] for q, _ in self._blocks)), -1)
        wpos[wcols] = np.arange(wcols.size)
        gathers = []
        for bi, (q, _ss) in enumerate(self._blocks):
            n_b = q.L_min.shape[1]
            local = wcols[(wcols >= offsets[bi]) & (wcols < offsets[bi] + n_b)]
            if local.size == 0:
                gathers.append(None)
                continue
            off_w = sum(q.field_size(v) for v in vars_[:iw])
            Pw = q.pre_right_pinv[local - offsets[bi]][:, off_w:off_w + q.field_size(t.field)]
            gathers.append((wpos[local], Pw))
        Z = np.zeros((wcols.size, refcols.size), dtype=self.dtype)
        block_of = np.searchsorted(offsets, refcols, side="right") - 1
        w = t.field
        for j, c in enumerate(refcols):
            q, ss = self._blocks[block_of[j]]
            for v in state:
                v["c"] = 0
            x = np.zeros(q.L_min.shape[1], dtype=self.dtype)
            x[c - offsets[block_of[j]]] = 1.0
            ss.scatter(q.pre_right @ x, state)
            X = t.op.evaluate()
            X.change_scales(1)
            Xg = X["g"]
            if t.ncc_first:
                vals = _pointwise(t.kind, t.ncc, t.rank_ncc, Xg, t.rank_op)
            else:
                vals = _pointwise(t.kind, Xg, t.rank_op, t.ncc, t.rank_ncc)
            w.change_scales(1)
            w["g"] = vals
            w["c"]
            for (q2, ss2), gth in zip(self._blocks, gathers):
                if gth is not None:
                    rows, Pw = gth
                    Z[rows, j] = Pw @ ss2.gather([w])
        return Z

    # ---- solver interface

    def assemble_evp_matrices(self, mass_equations=None):
        """(K, M) = (L_min, M_min); ``mass_equations`` is ignored (Dt defines M)."""
        return self.A_total, self.B_total

    def solve_evp(self, k: int = 6, sigma: Optional[complex] = None, return_eigenvectors: bool = False,
                  mass_equations=None, method: str = "arpack", factor: str = "scipy", **solver_options):
        from spdee.solvers import shift_invert
        from spdee.backends.fd2.compiled_problem import EvpResult
        if sigma is None:
            raise ValueError("the Dedalus backend needs a shift; pass sigma=")
        lam, V = shift_invert(self.A_total, self.B_total, sigma, k, method=method, factor=factor,
                              return_vectors=return_eigenvectors, **solver_options)
        lam = np.where(np.isfinite(lam), lam, np.complex128(np.nan))
        vectors = None
        if return_eigenvectors:
            vectors = [self.eigenvector_fields(V[:, j]) for j in range(V.shape[1])]
        return EvpResult(eigenvalues=lam, eigenvectors=vectors)

    def set_state(self, x_min: np.ndarray) -> None:
        """Load a pencil-space eigenvector into the Dedalus fields."""
        x = np.asarray(x_min).ravel()
        if self._T is not None:
            x = self._T @ x
        state = self.solver.state
        for v in state:
            v["c"] = 0
        for (q, ss), off in zip(self._blocks, self._offsets):
            n = q.L_min.shape[1]
            ss.scatter(q.pre_right @ x[off:off + n], state)

    def eigenvector_fields(self, x_min: np.ndarray) -> Dict[str, np.ndarray]:
        """Meridional grid data per unknown: scalars (N_theta, N_r), vectors
        (3, N_theta, N_r) in Dedalus component order (phi, theta, r)."""
        self.set_state(x_min)
        return {name: np.array(self.fields[name]["g"])[..., 0, :, :] for name in self.unknown_names}

    def __repr__(self) -> str:
        return (f"DedalusCompiledProblem(m={self.m}, N={self.A_total.shape[0]}, grid={self.grid}, "
                f"unknowns={self.unknown_names}, taus={[t.name for t in self.tau_fields]})")


def compile_dedalus(system: EquationSystem, grid, m: int, params: Optional[dict] = None,
                    ncc_cutoff: float = 1e-8, comm=None, **unused) -> DedalusCompiledProblem:
    if unused:
        warnings.warn(f"compile_dedalus: ignoring options {sorted(unused)}", RuntimeWarning)
    return DedalusCompiledProblem(system, grid, m, params=params, ncc_cutoff=ncc_cutoff, comm=comm)
