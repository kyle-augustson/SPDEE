"""IR -> spin-weighted spherical harmonics in (theta, phi) at fixed m times
finite differences in r.

Angular structure is analytic (``angular.py``), so every angular derivative
is a coefficient over l and the only discretization is the radial stencil
(Fornberg weights of selectable order).  Backgrounds that depend on theta
couple the l's through exact-quadrature multiplication matrices, one per
radial node.

State layout: unknowns in system order; a scalar occupies one block of
``nl*Nr`` coefficients, a vector three blocks in spin order (-, +, r) with
``v_- = (v_theta - i v_phi)/sqrt2``, ``v_+ = (v_theta + i v_phi)/sqrt2``,
a tensor nine blocks ``T^{ab}``, a and b in (-, +, r), block ``3*ia + ib``
of total spin ``s_a + s_b``; within a block the index is ``a_l*Nr + i``
(l-major, r inner).  Components whose spin exceeds l (spin pm1 at l = 0,
spin pm2 at l < 2) do not exist and are pinned.

Supported IR nodes: FieldRef, Sum, Negation, ScalarMultiple (background
times unknown, either order), Grad (scalar and vector), Div (vector and
tensor), Curl (vector), Lap (scalar and vector), LapH (scalar), Transpose,
Cross (background with unknown, either order), Dot (background vector with
unknown vector or tensor; background tensor with unknown vector, either
order), Dt on any unknown.  Boundary conditions are scalar rows at the
radial walls translated directly and imposed by row replacement on the
wall node of the component they act on most strongly; a composite row
(``BC_LINEAR_COMBINATION``: Robin, Navier slip, the normal stress) goes on
the host the walker's record names, the highest-rank unknown with a
theta_hat row on the spin + block and a phi_hat row on the spin - block.
A theta / phi pair of rows imposes the two spin components of one
tangential condition exactly; a lone tangential row is its Galerkin
projection on the spin-0 harmonics.

The constraint classifier (``spdee.ir.constraints.classify``, non-mimetic)
is consulted when the system is in its vocabulary: a tensor definition
``T - tensor(u) = 0`` is eliminated at compile time, a written ``Div(b) = 0``
with a curl evolution and no multiplier gets the divergence-cleaning
multiplier ``__div_b`` (``+ Grad`` of it in the evolution, Dirichlet zero at
the walls), and a gauge scalar present through Grad alone (psi of a vector
potential) puts its wall rows on its constraint equation, the divergence
row at the wall node.  Declared gauge unknowns get the mean-zero row at
m = 0 on their constraint equation.

Ball (``r_inner = 0``, ``docs/ball_sh_fd.md``): cell-centred nodes
``r_i = (i + 1/2) h`` with the wall as the last node, and the radial
stencils closed across the origin by mirrored ghost nodes carrying the
analytic continuation of the coefficients, ``(-1)^(l + n_r)`` with every
pm index mirrored (``f_l(-r) = (-1)^l f_l(r)`` for a scalar,
``u_r(-r) = (-1)^(l+1) u_r(r)`` and ``u_pm(-r) = (-1)^l u_mp(r)`` for a
vector, the same per index for a tensor).  Boundary conditions apply at
the outer wall only.

Pencil: ``K x = lambda M x`` with K the left-hand side without the Dt terms
and M the Dt coefficients (``omega = -i lambda``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import sympy

from spdee.ir.expressions import (
    Expression, FieldRef, Sum, Negation, ScalarMultiple, AsSymmetric, _BackgroundScalarValue,
)
from spdee.ir.operators import Grad, Div, Curl, Lap, LapH, Dt, Cross, Dot, Transpose
from spdee.ir.equations import Boundary, EquationSystem
from spdee.ir.fields import r as R_SYM, theta as TH_SYM
from spdee.ir.backgrounds import is_sympy_component, get_sympy_expr
from spdee.ir.constraints import classify
from spdee.ir.tasks import BC_LINEAR_COMBINATION, BC_LOC_R_INNER
from spdee.ir.walker import walk_equation_system, walk_boundary_conditions
from spdee.backends.sh_fd.angular import AngularVasil, sYlm_theta, kcoef

__all__ = ["ShFdGrid", "ShFdCompiledProblem", "compile_sh_fd"]

COMP_SPIN = (-1, +1, 0)
NCOMP = {"scalar": 1, "vector": 3, "tensor": 9}
SQ2 = np.sqrt(2.0)
MIRROR = {0: 1, 1: 0, 2: 2}
# e_s . (r, theta, phi) for the spin basis e_pm = (e_theta pm i e_phi)/sqrt2
SPIN_BASIS = {0: (1.0, 0.0, 0.0), +1: (0.0, 1 / SQ2, 1j / SQ2), -1: (0.0, 1 / SQ2, -1j / SQ2)}
# the spin block hosting a composite row of host_sigma
SIGMA_BLOCK = {-1: 0, +1: 1, 0: 2}


def fornberg_weights(z, x, m):
    x = np.asarray(x, float); n = len(x)
    C = np.zeros((n, m + 1)); c1 = 1.0; c4 = x[0] - z; C[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m); c2 = 1.0; c5 = c4; c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]; c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    C[i, k] = c1 * (k * C[i - 1, k - 1] - c5 * C[i - 1, k]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                C[j, k] = (c4 * C[j, k] - k * C[j, k - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2
    return C


def radial_diff_matrices(r, order):
    """First and second derivative matrices on the nodes r."""
    Nr = len(r); half = order // 2; width = order + 1
    if width > Nr:
        raise ValueError("fd_order + 1 must not exceed N_r")
    D1 = sp.lil_matrix((Nr, Nr)); D2 = sp.lil_matrix((Nr, Nr))
    for i in range(Nr):
        lo = min(max(i - half, 0), Nr - width)
        idx = np.arange(lo, lo + width)
        C = fornberg_weights(r[i], r[idx], 2)
        D1[i, idx] = C[:, 1]; D2[i, idx] = C[:, 2]
    return D1.tocsr(), D2.tocsr()


def radial_diff_matrices_ball(r, order):
    """Derivative matrices on the nodes r > 0 with stencils taken on the
    grid extended by the mirrored ghosts -r: (A1, A2) hold the weights on
    the nodes, (G1, G2) the ghost weights folded onto the mirrored node, so
    that D = A + sigma G differentiates a coefficient of parity sigma."""
    Nr = len(r); half = order // 2; width = order + 1
    if width > Nr:
        raise ValueError("fd_order + 1 must not exceed N_r")
    x = np.concatenate([-r[::-1], r])
    A1 = sp.lil_matrix((Nr, Nr)); A2 = sp.lil_matrix((Nr, Nr))
    G1 = sp.lil_matrix((Nr, Nr)); G2 = sp.lil_matrix((Nr, Nr))
    for i in range(Nr):
        lo = min(Nr + i - half, 2 * Nr - width)
        idx = np.arange(lo, lo + width)
        C = fornberg_weights(r[i], x[idx], 2)
        for k, w1, w2 in zip(idx, C[:, 1], C[:, 2]):
            if k >= Nr:
                A1[i, k - Nr] += w1; A2[i, k - Nr] += w2
            else:
                G1[i, Nr - 1 - k] += w1; G2[i, Nr - 1 - k] += w2
    return A1.tocsr(), A2.tocsr(), G1.tocsr(), G2.tocsr()


@dataclass
class ShFdGrid:
    """Radial nodes and angular truncation; ``r_inner = 0`` selects the ball
    (cell-centred nodes with the wall as the last node)."""
    N_r: int
    ell_max: int
    r_inner: float
    r_outer: float
    fd_order: int = 4
    nodes: Optional[np.ndarray] = None
    nquad: Optional[int] = None

    def __post_init__(self):
        if self.r_inner < 0:
            raise ValueError("r_inner must be >= 0")
        if self.fd_order % 2:
            raise ValueError("fd_order must be even")
        self.ball = self.r_inner == 0
        if self.nodes is not None:
            self.r = np.asarray(self.nodes, float)
            if self.ball and (self.r.min() <= 0 or not np.isclose(self.r[-1], self.r_outer)):
                raise ValueError("ball nodes must be positive and end at r_outer")
        elif self.ball:
            self.r = (np.arange(self.N_r) + 0.5) * self.r_outer / (self.N_r - 0.5)
        else:
            self.r = np.linspace(self.r_inner, self.r_outer, self.N_r)
        self.N_r = len(self.r)


class _Lin:
    """A linear operator on the unknowns: ``kind`` (scalar/vector/tensor),
    output size and blocks {unknown name: sparse (n_out x n_unknown)}."""

    def __init__(self, kind: str, blocks: Dict[str, sp.spmatrix], n_out: int):
        self.kind = kind
        self.blocks = blocks
        self.n_out = n_out

    def __add__(self, other: "_Lin") -> "_Lin":
        if self.kind != other.kind:
            raise TypeError("kind mismatch in sum")
        out = dict(self.blocks)
        for k, v in other.blocks.items():
            out[k] = (out[k] + v) if k in out else v
        return _Lin(self.kind, out, self.n_out)

    def scale(self, c) -> "_Lin":
        return _Lin(self.kind, {k: c * v for k, v in self.blocks.items()}, self.n_out)

    def apply(self, L: sp.spmatrix, kind: str, n_out: int) -> "_Lin":
        return _Lin(kind, {k: (L @ v).tocsr() for k, v in self.blocks.items()}, n_out)


class _Bg:
    """A background quantity on the (r, theta_quad) grid: scalar -> array
    (Nr, Nq); vector -> dict spin -> array; tensor -> dict (ia, ib) -> array
    of the spin components T^{ab}."""

    def __init__(self, kind: str, data):
        self.kind = kind
        self.data = data

    def map(self, fn, *others: "_Bg") -> "_Bg":
        """``fn`` applied entry by entry to this quantity and ``others``."""
        if self.kind == "scalar":
            return _Bg("scalar", fn(self.data, *[o.data for o in others]))
        return _Bg(self.kind, {k: fn(v, *[o.data[k] for o in others])
                               for k, v in self.data.items()})


class ShFdCompiledProblem:
    discretization = "sh_fd"

    def __init__(self, system: EquationSystem, grid, m: int, params: Optional[dict] = None):
        if not isinstance(grid, ShFdGrid):
            raise TypeError("sh_fd needs a ShFdGrid")
        self.system = system
        self.grid = grid
        self.m = int(m)
        self.params = {sympy.sympify(k): v for k, v in (params or {}).items()}
        self.ang = AngularVasil(self.m, grid.ell_max)
        self.ls = self.ang.ls; self.nl = self.ang.nl
        self.r = grid.r; self.Nr = len(self.r)
        self.nlr = self.nl * self.Nr
        self.ball = grid.ball
        if self.ball:
            self.D1, self.D2, self.G1, self.G2 = radial_diff_matrices_ball(self.r, grid.fd_order)
        else:
            self.D1, self.D2 = radial_diff_matrices(self.r, grid.fd_order)
        self.theta_q, self.w_q = self.ang.quadrature(grid.nquad)
        self.Nq = len(self.theta_q)
        self._Y = {s: np.array([sYlm_theta(s, l, self.m, self.theta_q) for l in self.ls])
                   for s in (-2, -1, 0, 1, 2)}
        self._Rq, self._Tq = np.meshgrid(self.r, self.theta_q, indexing="ij")
        self._mult_cache: Dict[Tuple[int, int, int], sp.spmatrix] = {}
        self._bg_cache: Dict[str, _Bg] = {}
        self._elim_cache: Dict[str, _Lin] = {}

        self.classification, self.classification_error = self._classify()
        self._eliminated = self._definitions()
        self._cleaning = {}
        if self.classification is not None:
            self._cleaning = {b: dc for b, dc in self.classification.divergence_cleaning.items()
                              if dc.constraint is not None}
        self._constraint_of = {name: eq for name, eq in (
            self.classification.multipliers.items() if self.classification else ())}
        for dc in self._cleaning.values():
            self._constraint_of[dc.multiplier] = dc.constraint

        self.layout: List[Tuple[str, str, int, int]] = []
        self.unknown_names = []
        off = 0
        kinds = [(name, f.kind) for name, f in system.unknowns.items()
                 if name not in self._eliminated]
        kinds += [(dc.multiplier, "scalar") for dc in self._cleaning.values()]
        for name, kind in kinds:
            size = NCOMP[kind] * self.nlr
            self.layout.append((name, kind, off, size))
            self.unknown_names.append(name)
            off += size
        self.N = off
        self._offset = {name: o for name, _k, o, _s in self.layout}
        self._kind = {name: k for name, k, _o, _s in self.layout}

        rows_K, rows_M = [], []
        self._eq_row_offset = {}
        roff = 0
        skipped = {eq for eq in self._eliminated_equations()}
        for eq in system.equations:
            if eq.name in skipped:
                continue
            K, M = self._equation_rows(eq)
            rows_K.append(K); rows_M.append(M)
            self._eq_row_offset[eq.name] = roff
            roff += K.shape[0]
        if roff != self.N:
            raise ValueError(f"sh_fd: {roff} equation rows for {self.N} unknowns; the system is "
                             f"not square{self._classifier_note()}")
        K = sp.vstack(rows_K, format="lil"); M = sp.vstack(rows_M, format="lil")
        used = set()
        self._apply_boundary_conditions(K, M, used)
        self._apply_cleaning_walls(K, M, used)
        self._apply_gauge_rows(K, M, used)
        self._pin_absent_harmonics(K, M)
        self.A_total = K.tocsr(); self.B_total = M.tocsr()

    # ---- the classifier's record

    def _classify(self):
        """The constraint classification of the system, or None with the
        classifier's reason when the system is outside its vocabulary (the
        direct translation then pairs unknowns with equations by count)."""
        try:
            tasks = walk_equation_system(self.system, self.m)
            record = classify(self.system, tasks, geometry="ball" if self.ball else "shell",
                              mimetic=False, core="sh_fd")
        except (NotImplementedError, ValueError, TypeError) as exc:
            return None, str(exc)
        return record, None

    def _classifier_note(self) -> str:
        if self.classification_error is None:
            return ""
        return f" (the constraint classifier refused it: {self.classification_error})"

    def _definitions(self) -> Dict[str, Tuple[complex, Expression]]:
        """{tensor: (c, rest)} for every classified definition
        c T + rest(u) = 0, so that T = -rest / c is substituted wherever T
        appears; c must be a constant."""
        out = {}
        if self.classification is None:
            return out
        equations = {eq.name: eq for eq in self.system.equations}
        for T, eq_name in self.classification.definitions.items():
            c = 0.0
            rest = None
            for sign, term in self._terms(equations[eq_name].lhs):
                if T not in self._referenced(term):
                    signed = term if sign > 0 else Negation(term)
                    rest = signed if rest is None else Sum(rest, signed)
                    continue
                coefficient = 1.0
                if isinstance(term, ScalarMultiple):
                    scalar = term.scalar if term.scalar.is_background else term.operand
                    term = term.operand if term.scalar.is_background else term.scalar
                    value = sympy.sympify(scalar.value).subs(self.params) if isinstance(
                        scalar, _BackgroundScalarValue) else None
                    if value is None or value.free_symbols:
                        raise NotImplementedError(
                            f"sh_fd: the coefficient of {T!r} in the definition {eq_name!r} "
                            f"must be a constant to eliminate {T!r}")
                    coefficient = complex(value)
                if not (isinstance(term, FieldRef) and term.field.name == T):
                    raise NotImplementedError(
                        f"sh_fd: {T!r} enters its definition {eq_name!r} through {term!r}; "
                        f"only c * {T} is eliminated")
                c += sign * coefficient
            if c == 0 or rest is None:
                raise NotImplementedError(f"sh_fd: the definition {eq_name!r} does not fix {T!r}")
            out[T] = (c, rest)
        return out

    def _eliminated_equations(self):
        if self.classification is None:
            return []
        return [self.classification.definitions[T] for T in self._eliminated]

    def _eliminated_lin(self, name: str) -> _Lin:
        if name not in self._elim_cache:
            c, rest = self._eliminated[name]
            self._elim_cache[name] = self._lin(rest).scale(-1.0 / c)
        return self._elim_cache[name]

    def _terms(self, e: Expression):
        """Top-level (sign, term) pairs of a Sum / Negation tree."""
        out = []

        def rec(x, sign):
            if isinstance(x, Sum):
                rec(x.a, sign); rec(x.b, sign)
            elif isinstance(x, Negation):
                rec(x.operand, -sign)
            else:
                out.append((sign, x))
        rec(e, 1)
        return out

    # ---- blocks

    def _kron(self, op_l, op_r):
        return sp.kron(sp.csr_matrix(op_l), sp.csr_matrix(op_r), format="csr")

    def _diag_l(self, vals):
        return sp.diags(np.asarray(vals, dtype=complex))

    def _mult(self, g: np.ndarray, s_in: int, s_out: int) -> sp.spmatrix:
        """Multiplication by g(r, theta) mapping spin s_in -> s_out: block
        diagonal in r with dense (nl x nl) blocks."""
        g = np.asarray(g, dtype=complex)
        key = (hash(g.tobytes()), s_in, s_out)
        if key in self._mult_cache:
            return self._mult_cache[key]
        Yin, Yout = self._Y[s_in], self._Y[s_out]
        Nr = self.Nr
        rows, cols, vals = [], [], []
        const = np.allclose(g, g[:1, :])
        Mi = None
        for i in range(Nr):
            if not const or Mi is None:
                Mi = 2 * np.pi * (Yout * self.w_q) @ (g[i] * Yin).T
            Mi_ = np.where(np.abs(Mi) > 1e-15, Mi, 0.0)
            jj, kk = np.nonzero(Mi_)
            rows.append(jj * Nr + i); cols.append(kk * Nr + i); vals.append(Mi_[jj, kk])
        if rows:
            A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.nlr, self.nlr)).tocsr()
        else:
            A = sp.csr_matrix((self.nlr, self.nlr), dtype=complex)
        self._mult_cache[key] = A
        return A

    def _eval_component(self, comp) -> np.ndarray:
        if is_sympy_component(comp):
            expr = sympy.sympify(get_sympy_expr(comp)).subs(self.params)
            free = expr.free_symbols - {R_SYM, TH_SYM}
            if free:
                raise ValueError(f"background {expr} has unbound symbols {free}; pass params")
            fn = sympy.lambdify((R_SYM, TH_SYM), expr, modules="numpy")
            val = fn(self._Rq, self._Tq)
            return np.broadcast_to(np.asarray(val, dtype=complex), self._Rq.shape).copy()
        return np.asarray(comp.evaluate_on_grid(self._Rq, self._Tq), dtype=complex) + 0.0 * self._Tq

    def _bg_field(self, f) -> _Bg:
        if f.name in self._bg_cache:
            return self._bg_cache[f.name]
        if f.kind == "scalar":
            out = _Bg("scalar", self._eval_component(f.component))
        elif f.kind == "vector":
            vr = self._eval_component(f.components["v_r"])
            vt = self._eval_component(f.components["v_theta"])
            vp = self._eval_component(f.components["v_phi"])
            out = _Bg("vector", {-1: (vt - 1j * vp) / SQ2, +1: (vt + 1j * vp) / SQ2, 0: vr})
        else:
            names = ("r", "theta", "phi")
            phys = {key: self._eval_component(comp) for key, comp in f.components.items()}
            data = {}
            for ia, sa in enumerate(COMP_SPIN):
                for ib, sb in enumerate(COMP_SPIN):
                    data[(ia, ib)] = sum(SPIN_BASIS[sa][i] * SPIN_BASIS[sb][j]
                                         * phys[(names[i], names[j])]
                                         for i in range(3) for j in range(3))
            out = _Bg("tensor", data)
        self._bg_cache[f.name] = out
        return out

    def _eval_bg(self, e: Expression) -> _Bg:
        """A background-only expression on the grid."""
        if isinstance(e, FieldRef):
            return self._bg_field(e.field)
        if isinstance(e, _BackgroundScalarValue):
            c = sympy.sympify(e.value).subs(self.params)
            if c.free_symbols - {R_SYM, TH_SYM}:
                raise ValueError(f"coefficient {c} has unbound symbols; pass params")
            if c.free_symbols:
                fn = sympy.lambdify((R_SYM, TH_SYM), c, modules="numpy")
                val = np.asarray(fn(self._Rq, self._Tq), dtype=complex)
                return _Bg("scalar", np.broadcast_to(val, self._Rq.shape).copy())
            return _Bg("scalar", np.full(self._Rq.shape, complex(c)))
        if isinstance(e, Sum):
            return self._eval_bg(e.a).map(np.add, self._eval_bg(e.b))
        if isinstance(e, Negation):
            return self._eval_bg(e.operand).map(np.negative)
        if isinstance(e, ScalarMultiple):
            a = self._eval_bg(e.scalar)
            return self._eval_bg(e.operand).map(lambda x: a.data * x)
        if isinstance(e, Transpose):
            a = self._eval_bg(e.operand)
            return _Bg("tensor", {(ia, ib): a.data[(ib, ia)] for (ia, ib) in a.data})
        if isinstance(e, Dot):
            a, b = self._eval_bg(e.a), self._eval_bg(e.b)
            if a.kind == "vector" and b.kind == "vector":
                return _Bg("scalar", a.data[0] * b.data[0] + a.data[1] * b.data[-1]
                           + a.data[-1] * b.data[1])
        if isinstance(e, Cross):
            a, b = self._eval_bg(e.a), self._eval_bg(e.b)
            X, B = a.data, b.data
            return _Bg("vector", {-1: 1j * (X[-1] * B[0] - X[0] * B[-1]),
                                  +1: 1j * (X[0] * B[1] - X[1] * B[0]),
                                  0: 1j * (X[1] * B[-1] - X[-1] * B[1])})
        raise NotImplementedError(f"sh_fd: cannot evaluate background expression {e!r}")

    def _dr_blocks(self, c0=None, c1=None, c2=None, comp=None, rank=1) -> Dict:
        """The radial operator c0 + c1 D1 + c2 D2 (coefficients None, a number
        or an r-diagonal) over all l on a scalar (``comp`` None) or on
        component ``comp`` of a rank-``rank`` field, as {target component:
        block}.  In the ball the ghost closure adds the folded stencil with
        the parity (-1)^(l + n_r), n_r the number of r indices, landing on
        the component with every pm index mirrored (u_pm -> u_mp)."""

        def combine(terms):
            out = None
            for c, D in terms:
                if c is None:
                    continue
                t = c @ D if sp.issparse(c) else (D if c == 1 else c * D)
                out = t if out is None else out + t
            return out

        I_r = sp.identity(self.Nr)
        main = self._kron(sp.identity(self.nl), combine([(c2, self.D2), (c1, self.D1), (c0, I_r)]))
        out = {comp: main}
        if self.ball:
            ghost = combine([(c2, self.G2), (c1, self.G1)])
            if ghost is not None:
                idx = () if comp is None else np.unravel_index(comp, (3,) * rank)
                sigma = (-1.0) ** (self.ls + sum(i == 2 for i in idx))
                mirror = None if comp is None else int(np.ravel_multi_index(
                    tuple(MIRROR[i] for i in idx), (3,) * rank))
                g = self._kron(self._diag_l(sigma), ghost)
                out[mirror] = (out[mirror] + g) if mirror in out else g
        return out

    def _identity(self, name: str) -> _Lin:
        kind = self._kind[name]
        n = NCOMP[kind] * self.nlr
        return _Lin(kind, {name: sp.identity(n, dtype=complex, format="csr")}, n)

    def _lin(self, e: Expression) -> _Lin:
        """The operator of an unknown-linear expression."""
        nlr = self.nlr
        Rinv = sp.diags(1.0 / self.r)
        if isinstance(e, FieldRef):
            if not e.field.is_unknown:
                raise TypeError("background where an unknown-linear expression was expected")
            if e.field.name in self._eliminated:
                return self._eliminated_lin(e.field.name)
            return self._identity(e.field.name)
        if isinstance(e, Sum):
            return self._lin(e.a) + self._lin(e.b)
        if isinstance(e, Negation):
            return self._lin(e.operand).scale(-1.0)
        if isinstance(e, AsSymmetric):
            return self._lin(e.operand)
        if isinstance(e, Dt):
            raise ValueError("Dt inside a non-Dt term")
        if isinstance(e, ScalarMultiple):
            a, b = e.scalar, e.operand
            if a.is_background and not b.is_background:
                return self._multiply(self._eval_bg(a), self._lin(b))
            if b.is_background and not a.is_background:
                return self._multiply(self._eval_bg(b), self._lin(a))
            raise NotImplementedError("sh_fd: product of two unknown-linear expressions")
        if isinstance(e, Grad):
            X = self._lin(e.operand)
            if X.kind == "scalar":
                return X.apply(self._grad_scalar(), "vector", 3 * nlr)
            if X.kind == "vector":
                return X.apply(self._grad_vector(), "tensor", 9 * nlr)
            raise NotImplementedError("sh_fd: Grad of a tensor")
        if isinstance(e, Div):
            X = self._lin(e.operand)
            if X.kind == "vector":
                return X.apply(self._div_vector(), "scalar", nlr)
            return X.apply(self._div_tensor(), "vector", 3 * nlr)
        if isinstance(e, Curl):
            X = self._lin(e.operand)
            if X.kind != "vector":
                raise TypeError("Curl needs a vector")
            return X.apply(self._curl(), "vector", 3 * nlr)
        if isinstance(e, Lap):
            X = self._lin(e.operand)
            if X.kind == "scalar":
                L = self._dr_blocks(c1=2.0 * Rinv, c2=1)[None] + self._laph_scalar()
                return X.apply(L, "scalar", nlr)
            if X.kind == "vector":
                return X.apply(self._lap_vector(), "vector", 3 * nlr)
            raise NotImplementedError("sh_fd: Lap of a tensor")
        if isinstance(e, LapH):
            return self._lin(e.operand).apply(self._laph_scalar(), "scalar", nlr)
        if isinstance(e, Transpose):
            X = self._lin(e.operand)
            if X.kind != "tensor":
                raise TypeError("Transpose needs a tensor")
            P = sp.lil_matrix((9 * nlr, 9 * nlr))
            I = sp.identity(nlr)
            for ia in range(3):
                for ib in range(3):
                    P[(3 * ib + ia) * nlr:(3 * ib + ia + 1) * nlr,
                      (3 * ia + ib) * nlr:(3 * ia + ib + 1) * nlr] = I
            return X.apply(P.tocsr(), "tensor", 9 * nlr)
        if isinstance(e, Cross):
            if e.a.is_background and not e.b.is_background:
                return self._cross_X_B(self._lin(e.b), self._eval_bg(e.a)).scale(-1.0)
            if e.b.is_background and not e.a.is_background:
                return self._cross_X_B(self._lin(e.a), self._eval_bg(e.b))
            raise NotImplementedError("sh_fd: Cross of two unknown-linear expressions")
        if isinstance(e, Dot):
            if e.a.is_background and not e.b.is_background:
                return self._dot_B_X(self._eval_bg(e.a), self._lin(e.b), left=True)
            if e.b.is_background and not e.a.is_background:
                return self._dot_B_X(self._eval_bg(e.b), self._lin(e.a), left=False)
            raise NotImplementedError("sh_fd: Dot of two unknown-linear expressions")
        raise NotImplementedError(f"sh_fd: unsupported IR node {type(e).__name__}")

    def _laph_scalar(self) -> sp.spmatrix:
        """The angular Laplacian -l(l+1)/r^2 of a scalar."""
        return self._kron(self._diag_l(-self.ls * (self.ls + 1.0)), sp.diags(1.0 / self.r ** 2))

    def _grad_scalar(self) -> sp.spmatrix:
        gm = self.ang.grad_scalar(); Rinv = sp.diags(1.0 / self.r)
        return sp.vstack([self._kron(self._diag_l(gm[0]), Rinv),
                          self._kron(self._diag_l(gm[1]), Rinv),
                          self._dr_blocks(c1=1)[None]], format="csr")

    def _div_vector(self) -> sp.spmatrix:
        dv = self.ang.div_vector_ang(); Rinv = sp.diags(1.0 / self.r)
        return sp.hstack([self._kron(self._diag_l(dv[0]), Rinv),
                          self._kron(self._diag_l(dv[1]), Rinv),
                          self._dr_blocks(c0=2.0 * Rinv, c1=1, comp=2)[2]], format="csr")

    def _curl(self) -> sp.spmatrix:
        nlr = self.nlr
        Rinv = sp.diags(1.0 / self.r)
        km0 = self._diag_l([kcoef(-1, l, 0) for l in self.ls])
        kp0 = self._diag_l([kcoef(+1, l, 0) for l in self.ls])
        kpm1 = self._diag_l([kcoef(+1, l, -1) for l in self.ls])
        kmp1 = self._diag_l([kcoef(-1, l, +1) for l in self.ls])
        Z = sp.csr_matrix((nlr, nlr))
        dm = self._dr_blocks(c0=Rinv, c1=1, comp=0); dp = self._dr_blocks(c0=Rinv, c1=1, comp=1)
        return sp.bmat([[-1j * dm[0], -1j * dm[1] if 1 in dm else Z, 1j * self._kron(km0, Rinv)],
                        [1j * dp[0] if 0 in dp else Z, 1j * dp[1], -1j * self._kron(kp0, Rinv)],
                        [1j * self._kron(kpm1, Rinv), -1j * self._kron(kmp1, Rinv), Z]],
                       format="csr")

    def _lap_vector(self) -> sp.spmatrix:
        """Vector Laplacian: radial Laplacian on each component minus Gamma/r^2,
        components (-, +, r) mapped to the (-, 0, +) order of Gamma."""
        Rinv = sp.diags(1.0 / self.r); Rinv2 = sp.diags(1.0 / self.r ** 2)
        nlr = self.nlr
        blocks = [[None] * 3 for _ in range(3)]
        vmap = {0: 0, 1: 2, 2: 1}
        for fi in range(3):
            lap = self._dr_blocks(c1=2.0 * Rinv, c2=1, comp=fi)
            for fj in range(3):
                vi, vj = vmap[fi], vmap[fj]
                gvals = []
                for l in self.ls:
                    L = l * (l + 1.0); s = np.sqrt(2 * L)
                    Gam = np.array([[L, -s, 0.0], [-s, L + 2.0, s], [0.0, s, L]])
                    gvals.append(Gam[vi, vj])
                G = self._kron(self._diag_l(gvals), Rinv2)
                if fi == fj:
                    blocks[fi][fj] = lap[fj] - G
                else:
                    blocks[fi][fj] = -G if np.any(np.abs(gvals) > 1e-13) else \
                        sp.csr_matrix((nlr, nlr))
                    if fj in lap:
                        blocks[fi][fj] = blocks[fi][fj] + lap[fj]
        return sp.bmat(blocks, format="csr")

    def _grad_vector(self) -> sp.spmatrix:
        """Grad(u) as the 9 spin components T^{ab} (a derivative index, b field
        component, both in the order (-, +, r); row block 3*ia + ib):
        T^{0,b} = d_r u_b; T^{mu,sigma} = (1/r) k^mu_{l,sigma} u_sigma with the
        connection terms T^{mu,0} -= u_mu/r and T^{mu,-mu} += u_r/r."""
        nlr = self.nlr
        Rinv = sp.diags(1.0 / self.r); I_l = sp.identity(self.nl)
        Z = sp.csr_matrix((nlr, nlr))
        R = self._kron(I_l, Rinv)

        def kR(mu, s):
            return self._kron(self._diag_l([kcoef(mu, l, s) for l in self.ls]), Rinv)

        blocks = {}
        for b in range(3):
            row = [Z, Z, Z]
            for c, blk in self._dr_blocks(c1=1, comp=b).items():
                row[c] = blk
            blocks[(2, b)] = row
        for a, mu in ((0, -1), (1, +1)):
            for b, s in ((0, -1), (1, +1), (2, 0)):
                row = [Z, Z, Z]
                row[b] = kR(mu, s)
                if b == 2:
                    row[a] = row[a] - R
                if s == -mu:
                    row[2] = row[2] + R
                blocks[(a, b)] = row
        rows = [sp.hstack(blocks[(ia, ib)], format="csr") for ia in range(3) for ib in range(3)]
        return sp.vstack(rows, format="csr")

    def _div_tensor(self) -> sp.spmatrix:
        """Div(T) on the nine spin components of _grad_vector, contracting the
        derivative with the first index: (Div T)^b = d_r T^{rb} + 2 T^{rb}/r
        + (1/r)(k^-_{l,1+s_b} T^{+b} + k^+_{l,-1+s_b} T^{-b}) with the
        connection terms -(T^{+-} + T^{-+})/r on b = r and +T^{mu r}/r on
        b = mu, the per-index rule of _grad_vector applied to both indices."""
        nlr = self.nlr
        Rinv = sp.diags(1.0 / self.r)
        R = self._kron(sp.identity(self.nl), Rinv)
        Z = sp.csr_matrix((nlr, nlr))

        def kR(mu, s):
            return self._kron(self._diag_l([kcoef(mu, l, s) for l in self.ls]), Rinv)

        rows = []
        for b, s_b in enumerate(COMP_SPIN):
            row = [Z] * 9
            for c, blk in self._dr_blocks(c0=2.0 * Rinv, c1=1, comp=6 + b, rank=2).items():
                row[c] = row[c] + blk
            row[3 + b] = row[3 + b] + kR(-1, 1 + s_b)
            row[b] = row[b] + kR(+1, -1 + s_b)
            if b == 2:
                row[3] = row[3] - R
                row[1] = row[1] - R
            else:
                row[3 * b + 2] = row[3 * b + 2] + R
            rows.append(row)
        return sp.bmat(rows, format="csr")

    def _multiply(self, bg: _Bg, X: _Lin) -> _Lin:
        """Background times unknown: a scalar keeps the spins, a vector or
        tensor times a scalar unknown gives its own rank."""
        nlr = self.nlr
        if bg.kind == "scalar":
            L = sp.block_diag([self._mult(bg.data, s, s) for s in self._spins(X.kind)],
                              format="csr")
            return X.apply(L, X.kind, NCOMP[X.kind] * nlr)
        if bg.kind == "vector" and X.kind == "scalar":
            L = sp.vstack([self._mult(bg.data[s], 0, s) for s in COMP_SPIN], format="csr")
            return X.apply(L, "vector", 3 * nlr)
        if bg.kind == "tensor" and X.kind == "scalar":
            L = sp.vstack([self._mult(bg.data[(ia, ib)], 0, sa + sb)
                           for ia, sa in enumerate(COMP_SPIN) for ib, sb in enumerate(COMP_SPIN)],
                          format="csr")
            return X.apply(L, "tensor", 9 * nlr)
        raise NotImplementedError(f"sh_fd: {bg.kind} background times a {X.kind} unknown")

    def _spins(self, kind):
        if kind == "scalar":
            return [0]
        if kind == "vector":
            return list(COMP_SPIN)
        return [sa + sb for sa in COMP_SPIN for sb in COMP_SPIN]

    def _cross_X_B(self, X: _Lin, B: _Bg) -> _Lin:
        """X x B for a vector unknown X and a background B:
        (X x B)_- = i (X_- B_0 - X_0 B_-), (X x B)_+ = i (X_0 B_+ - X_+ B_0),
        (X x B)_r = i (X_+ B_- - X_- B_+)."""
        if X.kind != "vector":
            raise TypeError("Cross needs vectors")
        nlr = self.nlr
        Bd = B.data
        rows = [
            [1j * self._mult(Bd[0], -1, -1), None, -1j * self._mult(Bd[-1], 0, -1)],
            [None, -1j * self._mult(Bd[0], +1, +1), 1j * self._mult(Bd[+1], 0, +1)],
            [-1j * self._mult(Bd[+1], -1, 0), 1j * self._mult(Bd[-1], +1, 0), None],
        ]
        L = sp.bmat([[c if c is not None else sp.csr_matrix((nlr, nlr)) for c in row]
                     for row in rows], format="csr")
        return X.apply(L, "vector", 3 * nlr)

    def _dot_B_X(self, B: _Bg, X: _Lin, left: bool) -> _Lin:
        """B.X = B_0 X_0 + B_+ X_- + B_- X_+ for a vector X; for a tensor X,
        (B.X)^b = B_0 X^{0b} + B_+ X^{-b} + B_- X^{+b} (left) or
        (X.B)^a = X^{a0} B_0 + X^{a-} B_+ + X^{a+} B_- (right); for a tensor
        background T and a vector X, (T.X)^a = sum_b T^{ab} X_{-b} (left) or
        (X.T)^b = sum_a X_{-a} T^{ab} (right)."""
        nlr = self.nlr
        Bd = B.data
        if B.kind == "tensor":
            if X.kind != "vector":
                raise NotImplementedError(
                    f"sh_fd: tensor background dotted with a {X.kind} unknown")
            rows = []
            for out, s_out in enumerate(COMP_SPIN):
                row = [None] * 3
                for c, s_c in enumerate(COMP_SPIN):
                    key = (out, c) if left else (c, out)
                    row[MIRROR[c]] = self._mult(Bd[key], -s_c, s_out)
                rows.append(row)
            return X.apply(sp.bmat(rows, format="csr"), "vector", 3 * nlr)
        if X.kind == "vector":
            L = sp.hstack([self._mult(Bd[+1], -1, 0), self._mult(Bd[-1], +1, 0),
                           self._mult(Bd[0], 0, 0)], format="csr")
            return X.apply(L, "scalar", nlr)
        if X.kind == "tensor":
            pair = {0: Bd[+1], 1: Bd[-1], 2: Bd[0]}
            rows = []
            for out in range(3):
                s_out = COMP_SPIN[out]
                row = [None] * 9
                for c in range(3):
                    s_c = COMP_SPIN[c]
                    idx = (3 * c + out) if left else (3 * out + c)
                    row[idx] = self._mult(pair[c], s_c + s_out, s_out)
                rows.append([x if x is not None else sp.csr_matrix((nlr, nlr)) for x in row])
            return X.apply(sp.bmat(rows, format="csr"), "vector", 3 * nlr)
        raise NotImplementedError("sh_fd: Dot with a scalar")

    # ---- equations

    def _split_dt(self, e: Expression):
        """(Dt terms, other terms) as lists of (sign, expression)."""
        dt_terms, others = [], []
        for sign, x in self._terms(e):
            (dt_terms if self._contains_dt(x) else others).append((sign, x))
        return dt_terms, others

    def _contains_dt(self, e):
        return isinstance(e, Dt) or any(self._contains_dt(c) for c in e.children)

    def _dt_operator(self, e: Expression) -> _Lin:
        """The operator of coefficient * X for a term coefficient * Dt(X)."""
        if isinstance(e, Dt):
            return self._lin(e.operand)
        if isinstance(e, ScalarMultiple):
            if e.scalar.is_background:
                return self._multiply(self._eval_bg(e.scalar), self._dt_operator(e.operand))
            if e.operand.is_background:
                return self._multiply(self._eval_bg(e.operand), self._dt_operator(e.scalar))
        if isinstance(e, Negation):
            return self._dt_operator(e.operand).scale(-1.0)
        raise NotImplementedError(f"sh_fd: unsupported Dt term {e!r}")

    def _assemble(self, lin: _Lin) -> sp.spmatrix:
        cols = [lin.blocks[name] if name in lin.blocks else sp.csr_matrix((lin.n_out, size))
                for name, _kind, _off, size in self.layout]
        return sp.hstack(cols, format="csr")

    def _equation_rows(self, eq):
        dt_terms, others = self._split_dt(eq.lhs)
        n_out = NCOMP[eq.kind] * self.nlr
        K = sp.csr_matrix((n_out, self.N), dtype=complex)
        M = sp.csr_matrix((n_out, self.N), dtype=complex)
        for sign, t in others:
            K = K + sign * self._assemble(self._lin(t))
        for sign, t in dt_terms:
            M = M + sign * self._assemble(self._dt_operator(t))
        for b, dc in self._cleaning.items():
            if dc.evolution == eq.name:
                grad = _Lin("vector", {dc.multiplier: self._grad_scalar()}, 3 * self.nlr)
                K = K + self._assemble(grad)
        return K, M

    # ---- wall rows, gauge rows and absent harmonics

    def _spin_of(self, name: str, c: int) -> int:
        kind = self._kind[name]
        if kind == "scalar":
            return 0
        if kind == "vector":
            return COMP_SPIN[c]
        return COMP_SPIN[c // 3] + COMP_SPIN[c % 3]

    def _absent(self, name: str, c: int) -> int:
        """The number of leading l's at which component c of ``name`` does
        not exist (l < |spin|)."""
        return max(0, abs(self._spin_of(name, c)) - self.m)

    def _apply_boundary_conditions(self, K, M, used):
        """Replace, for each boundary row, the wall-node rows of the block it
        acts on most strongly; a composite row goes on the block the walker's
        record names (the highest-rank unknown, theta_hat on spin +, phi_hat
        on spin -)."""
        Nr = self.Nr
        try:
            tasks = walk_boundary_conditions(self.system, self.m, composite=True)
        except (NotImplementedError, ValueError, TypeError):
            tasks = [None] * len(self.system.boundary_conditions)
        for bc, task in zip(self.system.boundary_conditions, tasks):
            if bc.location in (Boundary.THETA_NORTH, Boundary.THETA_SOUTH):
                raise NotImplementedError("sh_fd: theta boundary conditions do not apply")
            if bc.location == Boundary.R_INNER and self.ball:
                raise ValueError(f"BC {bc.name!r} at the inner wall: a ball has no inner wall, "
                                 "regularity at the origin is built in")
            i_wall = 0 if bc.location == Boundary.R_INNER else Nr - 1
            lin = self._lin(bc.expr)
            if lin.kind != "scalar":
                raise TypeError("boundary conditions must be scalar")
            row_op = self._assemble(lin).tocsr()
            if task is not None and task.kind == BC_LINEAR_COMBINATION:
                name, c = self._record_host(task)
            else:
                name, c = self._strongest_component(bc, lin, row_op, i_wall, used)
            self._place_wall_row(K, M, row_op, name, c, i_wall, used, f"BC {bc.name!r}")

    def _record_host(self, task) -> Tuple[str, int]:
        """(host unknown, block) of a composite boundary row."""
        name = task.unknown_name
        if name in self._eliminated:
            raise NotImplementedError(f"sh_fd: BC {task.name!r} constrains the eliminated tensor "
                                      f"{name!r}")
        if self._kind[name] == "scalar":
            return name, 0
        return name, SIGMA_BLOCK[task.metadata["host_sigma"]]

    def _strongest_component(self, bc, lin, row_op, i_wall, used) -> Tuple[str, int]:
        """The free wall component the row acts on most strongly."""
        Nr, nlr = self.Nr, self.nlr
        candidates = []
        for name, kind, off, size in self.layout:
            if name not in lin.blocks:
                continue
            for c in range(NCOMP[kind]):
                if (name, c, i_wall) in used:
                    continue
                blk = lin.blocks[name][:, c * nlr:(c + 1) * nlr]
                strength = sum(np.abs(blk[:, a_l * Nr + i_wall].toarray()).sum()
                               for a_l in range(self.nl))
                candidates.append((strength, name, c))
        if not candidates:
            raise ValueError(f"BC {bc.name!r}: no free wall row to impose it on")
        strength, name, c = max(candidates)
        if strength == 0.0:
            raise ValueError(f"BC {bc.name!r} does not act on the wall node")
        return name, c

    def _wall_slot(self, name: str, c: int, i_wall: int) -> Tuple[str, int, tuple]:
        """(equation, block, key) whose wall-node rows host a wall row on
        component c of ``name``: a vector's own equation, a scalar's scalar
        equation, and for a scalar present through Grad alone (a multiplier)
        its constraint equation, whose rows are the multiplier's.  Handing a
        multiplier's row to the vector's radial block, as the tau method
        does, drops the radial evolution row at the wall node instead of the
        constraint row there and loses an order of convergence on the
        poloidal free-decay modes of the vector potential."""
        kind = self._kind[name]
        if kind == "tensor":
            raise NotImplementedError("sh_fd: boundary rows on a tensor unknown are not supported")
        if kind == "vector":
            return self._host_equation(name), c, (name, c, i_wall)
        eq = self._scalar_equation(name)
        if eq is not None:
            return eq, 0, (name, 0, i_wall)
        return self._constraint_equation(name), 0, (name, 0, i_wall)

    def _place_wall_row(self, K, M, row_op, name, c, i_wall, used, label):
        """Replace the wall-node rows over l of the slot hosting component c
        of ``name`` by the rows of ``row_op`` (nl*Nr x N)."""
        Nr, nlr = self.Nr, self.nlr
        eq, blk, key = self._wall_slot(name, c, i_wall)
        if key in used:
            raise ValueError(f"{label}: the wall row of {key[0]!r} (block {key[1]}) at node "
                             f"{i_wall} is already taken")
        used.add(key)
        hoff = self._eq_row_offset[eq]
        # a component absent at the leading l's has no slot there: its rows go on the slots above
        shift = self._absent(key[0], blk)
        for a_l in range(shift, self.nl):
            target = hoff + blk * nlr + a_l * Nr + i_wall
            src = (a_l - shift) * Nr + i_wall
            K.rows[target] = list(row_op.indices[row_op.indptr[src]:row_op.indptr[src + 1]])
            K.data[target] = list(row_op.data[row_op.indptr[src]:row_op.indptr[src + 1]])
            M.rows[target] = []
            M.data[target] = []

    def _apply_cleaning_walls(self, K, M, used):
        """Dirichlet zero on every divergence-cleaning multiplier at the walls
        the record names."""
        for b, dc in self._cleaning.items():
            row_op = self._assemble(self._identity(dc.multiplier)).tocsr()
            for wall in dc.walls:
                i_wall = 0 if wall == BC_LOC_R_INNER else self.Nr - 1
                self._place_wall_row(K, M, row_op, dc.multiplier, 0, i_wall, used,
                                     f"divergence cleaning of {b!r}")

    def _radial_weights(self) -> np.ndarray:
        """Quadrature weights of int f dr on the nodes: the trapezoid rule in
        the shell, the cell widths (edges at the midpoints, the origin and
        the wall) in the ball."""
        r = self.r
        if self.ball:
            edges = np.concatenate([[0.0], 0.5 * (r[1:] + r[:-1]), [r[-1]]])
            return np.diff(edges)
        w = np.zeros(self.Nr)
        w[:-1] += 0.5 * np.diff(r); w[1:] += 0.5 * np.diff(r)
        return w

    def _apply_gauge_rows(self, K, M, used):
        """The mean-zero row int p dV = 0 of every declared gauge unknown at
        m = 0, on the row of its constraint equation at the first node inside
        the outer wall: the l = 0 coefficient weighted by r^2.  A gauge
        unknown with a wall row of its own (psi = 0 of a vector potential)
        has no constant mode left to fix and gets no row: the mean-zero row
        would then replace a constraint row the vector needs."""
        if self.m != 0:
            return
        Nr = self.Nr
        w = self._radial_weights() * self.r ** 2
        for name in self.system.gauge_unknowns:
            if any(key[0] == name for key in used):
                continue
            target = self._eq_row_offset[self._constraint_equation(name)] + Nr - 2
            K.rows[target] = [self._offset[name] + i for i in range(Nr)]
            K.data[target] = [complex(x) for x in w]
            M.rows[target] = []
            M.data[target] = []

    def _pin_absent_harmonics(self, K, M):
        """A spin component at l < |spin| (spin pm1 at l = 0, spin pm2 at
        l < 2) does not exist: its rows in the host equation become the unit,
        with no Dt part."""
        Nr, nlr = self.Nr, self.nlr
        for name, kind, off, size in self.layout:
            if kind == "scalar":
                continue
            hoff = None
            for c in range(NCOMP[kind]):
                n = self._absent(name, c)
                if n == 0:
                    continue
                if hoff is None:
                    hoff = self._eq_row_offset[self._host_equation(name)]
                for i in range(n * Nr):
                    K.rows[hoff + c * nlr + i] = [off + c * nlr + i]
                    K.data[hoff + c * nlr + i] = [1.0]
                    M.rows[hoff + c * nlr + i] = []
                    M.data[hoff + c * nlr + i] = []

    def _host_equation(self, name: str) -> str:
        """The equation of the same kind carrying Dt of a vector or tensor
        unknown, else the first of that kind referencing it."""
        kind = self._kind[name]
        for eq in self.system.equations:
            if eq.kind == kind and eq.name in self._eq_row_offset and self._has_dt_of(eq.lhs, name):
                return eq.name
        for eq in self.system.equations:
            if eq.kind == kind and eq.name in self._eq_row_offset \
                    and name in self._referenced(eq.lhs):
                return eq.name
        raise ValueError(f"no equation hosts unknown {name!r}")

    def _scalar_equation(self, name: str) -> Optional[str]:
        """The scalar equation carrying Dt of ``name``, else the first
        scalar equation referencing it; None for a scalar present through
        Grad alone."""
        for eq in self.system.equations:
            if eq.kind == "scalar" and self._has_dt_of(eq.lhs, name):
                return eq.name
        for eq in self.system.equations:
            if eq.kind == "scalar" and name in self._referenced(eq.lhs):
                return eq.name
        return None

    def _constraint_equation(self, name: str) -> str:
        """The scalar Dt-free equation whose rows are a multiplier's: the
        classifier's, else the first not referencing ``name``, else any."""
        if name in self._constraint_of:
            return self._constraint_of[name]
        eqs = [eq for eq in self.system.equations
               if eq.kind == "scalar" and not self._contains_dt(eq.lhs)]
        for eq in eqs:
            if name not in self._referenced(eq.lhs):
                return eq.name
        if eqs:
            return eqs[0].name
        raise ValueError(f"no equation hosts unknown {name!r}{self._classifier_note()}")

    def _has_dt_of(self, e, name):
        if isinstance(e, Dt):
            return name in self._referenced(e.operand)
        return any(self._has_dt_of(c, name) for c in e.children)

    def _referenced(self, e) -> set:
        out = set()

        def rec(x):
            if isinstance(x, FieldRef) and x.field.is_unknown:
                out.add(x.field.name)
            for c in x.children:
                rec(c)
        rec(e)
        return out

    # ---- solve

    def assemble_evp_matrices(self, mass_equations=None):
        return self.A_total, self.B_total

    def solve_evp(self, k: int = 6, sigma: Optional[complex] = None,
                  return_eigenvectors: bool = False, mass_equations=None, method: str = "arpack",
                  factor: str = "scipy", **solver_options):
        from spdee.solvers import shift_invert
        from spdee.backends.fd2.compiled_problem import EvpResult
        if sigma is None:
            raise ValueError("the sh_fd backend needs a shift; pass sigma=")
        lam, V = shift_invert(self.A_total, self.B_total, sigma, k, method=method, factor=factor,
                              return_vectors=return_eigenvectors, **solver_options)
        lam = np.where(np.isfinite(lam), lam, np.complex128(np.nan))
        vectors = None
        if return_eigenvectors:
            vectors = [self.unpack(V[:, j]) for j in range(V.shape[1])]
        return EvpResult(eigenvalues=lam, eigenvectors=vectors)

    def unpack(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """{name: (ncomp, nl, Nr)} spectral coefficients (component order -, +, r;
        a tensor's nine blocks 3*ia + ib)."""
        out = {}
        for name, kind, off, size in self.layout:
            out[name] = np.asarray(x[off:off + size]).reshape(NCOMP[kind], self.nl, self.Nr)
        return out

    def to_grid(self, coeffs: np.ndarray, theta: np.ndarray, spin: int = 0) -> np.ndarray:
        """Synthesize one component (nl, Nr) on theta at phi = 0 -> (N_theta, Nr)."""
        Y = np.array([sYlm_theta(spin, l, self.m, theta) for l in self.ls])
        return Y.T @ coeffs

    def __repr__(self):
        return (f"ShFdCompiledProblem(m={self.m}, N={self.N}, N_r={self.Nr}, "
                f"ell_max={self.grid.ell_max}, fd_order={self.grid.fd_order}, "
                f"unknowns={self.unknown_names})")


def compile_sh_fd(system: EquationSystem, grid: ShFdGrid, m: int, params: Optional[dict] = None,
                  **unused) -> ShFdCompiledProblem:
    if unused:
        warnings.warn(f"compile_sh_fd: ignoring options {sorted(unused)}", RuntimeWarning)
    return ShFdCompiledProblem(system, grid, m, params=params)
