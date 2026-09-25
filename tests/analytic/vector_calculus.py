"""Symbolic spherical vector calculus at a fixed azimuthal order.

Fields carry the factor ``exp(i m phi)`` implicitly, so the azimuthal
derivative is multiplication by ``i m``. A scalar is a SymPy expression
in ``(r, theta)``, a vector a :class:`Vec` of physical components in the
orthonormal frame order ``(r, theta, phi)``, and a rank-2 tensor a
:class:`Ten` of ``T[i][j]`` in the same frame.

Every operator is built from one covariant derivative, which uses the
connection coefficients of the orthonormal spherical frame; the
component formulas of Grad, Div, Curl and the tensor divergence are
never written out by hand.

    Gamma[i][j][l] is the e_l coefficient of nabla_{e_i} e_j,
    (nabla v)_{i l} = D_i v_l + sum_j v_j Gamma[i][j][l].

``Dot(a, Grad(v)) = a_i (Grad v)_{ij}``: the IR contracts a vector with
the first index of a rank-2 tensor and ``(Grad v)_{ij} = nabla_i v_j``,
so ``Dot(B0, Grad(v))`` is the advection ``(B0 . nabla) v``.

:func:`namespace` returns the evaluation namespace that mirrors
``spdee.problem.DEFAULT_NAMESPACE``, so one operator string is evaluated
both here (giving the manufactured image) and in the IR (giving the
problem the cores compile).
"""

import sympy as sp

from spdee.ir import r, theta

AXES = (0, 1, 2)

_SIN = sp.sin(theta)
_COT = sp.cos(theta) / _SIN


class Vec:
    """A vector by its physical components (v_r, v_theta, v_phi)."""

    __slots__ = ("c",)

    def __init__(self, *comps):
        if len(comps) == 1:
            comps = tuple(comps[0])
        self.c = tuple(sp.sympify(x) for x in comps)

    def __getitem__(self, i):
        return self.c[i]

    def __add__(self, other):
        return Vec(a + b for a, b in zip(self.c, _as_vec(other).c))

    __radd__ = __add__

    def __sub__(self, other):
        return Vec(a - b for a, b in zip(self.c, _as_vec(other).c))

    def __rsub__(self, other):
        return Vec(b - a for a, b in zip(self.c, _as_vec(other).c))

    def __neg__(self):
        return Vec(-a for a in self.c)

    def __mul__(self, k):
        return Vec(a * k for a in self.c)

    __rmul__ = __mul__

    def __truediv__(self, k):
        return Vec(a / k for a in self.c)

    def __repr__(self):
        return f"Vec{self.c}"


class Ten:
    """A rank-2 tensor by its physical components T[i][j]."""

    __slots__ = ("c",)

    def __init__(self, rows):
        self.c = tuple(tuple(sp.sympify(x) for x in row) for row in rows)

    def __getitem__(self, i):
        return self.c[i]

    def __add__(self, other):
        return Ten([[a + b for a, b in zip(ra, rb)]
                    for ra, rb in zip(self.c, other.c)])

    __radd__ = __add__

    def __sub__(self, other):
        return Ten([[a - b for a, b in zip(ra, rb)]
                    for ra, rb in zip(self.c, other.c)])

    def __neg__(self):
        return Ten([[-a for a in row] for row in self.c])

    def __mul__(self, k):
        return Ten([[a * k for a in row] for row in self.c])

    __rmul__ = __mul__

    def __repr__(self):
        return f"Ten{self.c}"


def _as_vec(x):
    if isinstance(x, Vec):
        return x
    raise TypeError(f"expected a vector, got {x!r}")


def _gamma():
    """Connection coefficients of the orthonormal spherical frame."""
    g = [[[sp.S.Zero] * 3 for _ in AXES] for _ in AXES]
    g[1][1][0] = -1 / r
    g[1][0][1] = 1 / r
    g[2][2][0] = -1 / r
    g[2][2][1] = -_COT / r
    g[2][0][2] = 1 / r
    g[2][1][2] = _COT / r
    return g


GAMMA = _gamma()


def _deriv(expr, axis, m):
    """The frame derivative e_axis applied to a scalar expression."""
    if axis == 0:
        return sp.diff(expr, r)
    if axis == 1:
        return sp.diff(expr, theta) / r
    return sp.I * m * expr / (r * _SIN)


def grad(x, m):
    """Grad of a scalar (-> Vec) or of a vector (-> Ten)."""
    if isinstance(x, Vec):
        return Ten([[_deriv(x[j], i, m) + sum(x[k] * GAMMA[i][k][j] for k in AXES)
                     for j in AXES] for i in AXES])
    return Vec(_deriv(x, i, m) for i in AXES)


def div(x, m):
    """Div of a vector (-> scalar) or of a tensor on its first index."""
    if isinstance(x, Ten):
        out = []
        for b in AXES:
            acc = sp.S.Zero
            for i in AXES:
                acc += _deriv(x[i][b], i, m)
                acc += sum(x[k][b] * GAMMA[i][k][i] for k in AXES)
                acc += sum(x[i][k] * GAMMA[i][k][b] for k in AXES)
            out.append(acc)
        return Vec(out)
    g = grad(x, m)
    return sum(g[i][i] for i in AXES)


def curl(v, m):
    """(curl v)_k = eps_{kij} nabla_i v_j."""
    g = grad(v, m)
    return Vec(g[1][2] - g[2][1], g[2][0] - g[0][2], g[0][1] - g[1][0])


def lap(x, m):
    """Lap of a scalar, or the vector Laplacian grad div - curl curl."""
    if isinstance(x, Vec):
        gd, cc = grad(div(x, m), m), curl(curl(x, m), m)
        return Vec(gd[i] - cc[i] for i in AXES)
    return div(grad(x, m), m)


def laph(s, m):
    """The angular part of the scalar Laplacian."""
    return lap(s, m) - sp.diff(r ** 2 * sp.diff(s, r), r) / r ** 2


def transpose(T):
    return Ten([[T[j][i] for j in AXES] for i in AXES])


def trace(T):
    return sum(T[i][i] for i in AXES)


def dot(a, b):
    """Vector-vector, tensor-vector, vector-tensor and tensor-tensor."""
    if isinstance(a, Vec) and isinstance(b, Vec):
        return sum(a[i] * b[i] for i in AXES)
    if isinstance(a, Ten) and isinstance(b, Vec):
        return Vec(sum(a[i][j] * b[j] for j in AXES) for i in AXES)
    if isinstance(a, Vec) and isinstance(b, Ten):
        return Vec(sum(a[i] * b[i][j] for i in AXES) for j in AXES)
    return Ten([[sum(a[i][k] * b[k][j] for k in AXES) for j in AXES] for i in AXES])


def cross(a, b):
    return Vec(a[1] * b[2] - a[2] * b[1],
               a[2] * b[0] - a[0] * b[2],
               a[0] * b[1] - a[1] * b[0])


R_HAT, THETA_HAT, PHI_HAT = Vec(1, 0, 0), Vec(0, 1, 0), Vec(0, 0, 1)


def namespace(m, extra=None):
    """The evaluation namespace mirroring ``DEFAULT_NAMESPACE``."""
    ns = {
        "grad": lambda x: grad(x, m), "Grad": lambda x: grad(x, m),
        "div": lambda x: div(x, m), "Div": lambda x: div(x, m),
        "curl": lambda x: curl(x, m), "Curl": lambda x: curl(x, m),
        "lap": lambda x: lap(x, m), "Lap": lambda x: lap(x, m),
        "laph": lambda x: laph(x, m), "LapH": lambda x: laph(x, m),
        "trans": transpose, "transpose": transpose, "Transpose": transpose,
        "trace": trace, "Trace": trace,
        "dot": dot, "Dot": dot, "cross": cross, "Cross": cross,
        "r_hat": R_HAT, "theta_hat": THETA_HAT, "phi_hat": PHI_HAT,
        "r": r, "theta": theta,
    }
    for name in ("sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "pi", "I"):
        ns[name] = getattr(sp, name)
    ns.update(extra or {})
    return ns
