"""State-augmentation pass for composed unknown operators.

The walker requires Cross to receive bare FieldRefs at its leaves. The
composed forms

    Cross(Curl(B), B0)        # Lorentz term
    Curl(Cross(v, B0))        # induction term

(and their bg-first variants) can instead be unfolded with an auxiliary
unknown:

    Cross(Curl(B), B0)
        -> aux_J = Curl(B)         # defining equation: aux_J - Curl(B) = 0
        -> Cross(aux_J, B0)

The defining equation joins the bulk equation list and the auxiliary
becomes an ordinary unknown with its own row and column block.
Structurally identical inner expressions share one auxiliary.

`augment_equation_system(system, prefer_aux_free=True)` leaves both
composed forms untouched, because the walker emits them directly as
TASK_CROSS_CURL_UNK_BG / TASK_CURL_CROSS_UNK_BG. With
`prefer_aux_free=False` the auxiliary rewrite above is applied. In
either mode, a composed term whose background vector is identically
zero is dropped from its sum; the drop propagates through every
operator applied to it (Div(0) = 0, ...), and an equation whose whole
left-hand side drops raises ValueError.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import sympy as sp

from spdee.ir.expressions import Expression, FieldRef, Sum, Negation, ScalarMultiple
from spdee.ir.operators import Curl, Cross, Grad, Div, Dot, Lap, LapH, Dt
from spdee.ir.fields import VectorField, FieldRole
from spdee.ir.equations import Equation, EquationSystem


# ---------------------------------------------------------------------------
# Site detection
# ---------------------------------------------------------------------------

def _is_bare_vec_fieldref(expr: Expression, background: bool) -> bool:
    """True if expr is FieldRef(vector field) with the given background flag."""
    return (
        isinstance(expr, FieldRef)
        and expr.field.kind == "vector"
        and expr.field.is_background == background
    )


def _is_unknown_curl_of_vec(expr: Expression) -> bool:
    """True if expr is Curl(FieldRef(unknown vector))."""
    return (
        isinstance(expr, Curl)
        and _is_bare_vec_fieldref(expr.operand, background=False)
    )


def _is_cross_unknown_bg(expr: Expression) -> bool:
    """True if expr is Cross(unknown_vec, bg_vec) or Cross(bg_vec, unknown_vec)
    with both operands bare FieldRefs."""
    if not isinstance(expr, Cross):
        return False
    a, b = expr.a, expr.b
    return (
        (_is_bare_vec_fieldref(a, False) and _is_bare_vec_fieldref(b, True))
        or (_is_bare_vec_fieldref(a, True) and _is_bare_vec_fieldref(b, False))
    )


def _is_identically_zero_bg_vector(field_ref: FieldRef) -> bool:
    """True if `field_ref` is a background vector whose three SymPy
    components all simplify to 0.

    Conservative: numeric or otherwise non-SymPy components return
    False, so the term is kept.
    """
    fld = field_ref.field
    if not fld.is_background:
        return False
    if fld.kind != "vector":
        return False
    for comp in (fld.v_r, fld.v_theta, fld.v_phi):
        try:
            simplified = sp.simplify(comp)
        except (sp.SympifyError, TypeError, AttributeError):
            return False
        if simplified != 0:
            return False
    return True


def _expr_signature(expr: Expression) -> Tuple:
    """Hashable structural signature of an Expression, so that identical
    compositions share one auxiliary. FieldRefs are keyed by field name
    (unique within an EquationSystem)."""
    if isinstance(expr, FieldRef):
        return ("FieldRef", expr.field.name)
    if isinstance(expr, Curl):
        return ("Curl", _expr_signature(expr.operand))
    if isinstance(expr, Div):
        return ("Div", _expr_signature(expr.operand))
    if isinstance(expr, Grad):
        return ("Grad", _expr_signature(expr.operand))
    if isinstance(expr, Cross):
        return ("Cross", _expr_signature(expr.a), _expr_signature(expr.b))
    if isinstance(expr, Dot):
        return ("Dot", _expr_signature(expr.a), _expr_signature(expr.b))
    if isinstance(expr, Lap):
        return ("Lap", _expr_signature(expr.operand))
    if isinstance(expr, Dt):
        return ("Dt", _expr_signature(expr.operand))
    if isinstance(expr, Negation):
        return ("Neg", _expr_signature(expr.operand))
    if isinstance(expr, Sum):
        return ("Sum", tuple(_expr_signature(a) for a in expr.children))
    if isinstance(expr, ScalarMultiple):
        return ("Mul",
                _expr_signature(expr.scalar),
                _expr_signature(expr.operand))
    return ("opaque", repr(expr))


# ---------------------------------------------------------------------------
# Augmentation state
# ---------------------------------------------------------------------------

class _AugContext:
    """Mutable bookkeeping during the augmentation rewrite."""

    def __init__(self, prefer_aux_free: bool = True):
        # signature -> auxiliary unknown field
        self.cache: Dict[Tuple, VectorField] = {}
        self.new_unknowns: List[VectorField] = []
        self.new_equations: List[Equation] = []
        self._counter = 0
        # True: leave Cross(Curl(unknown), bg) and Curl(Cross(unknown, bg))
        # to the walker's composed dispatchers. False: unfold them with
        # auxiliary unknowns.
        self.prefer_aux_free = prefer_aux_free

    def _new_aux(self, aux_name: str, inner: Expression) -> FieldRef:
        """Create the auxiliary unknown `aux_name` with defining equation
        aux - inner = 0 and return a FieldRef to it."""
        aux_field = VectorField(name=aux_name, role=FieldRole.UNKNOWN)
        self.cache[_expr_signature(inner)] = aux_field
        self.new_unknowns.append(aux_field)
        aux_ref = FieldRef(aux_field)
        self.new_equations.append(
            Equation(lhs=aux_ref + (-inner), name=f"__def_{aux_name}")
        )
        return aux_ref

    def get_or_create_aux_for_curl(self, inner: Curl) -> FieldRef:
        """Auxiliary unknown for `inner = Curl(unknown_vec)`."""
        sig = _expr_signature(inner)
        if sig in self.cache:
            return FieldRef(self.cache[sig])
        operand = inner.operand
        assert isinstance(operand, FieldRef)
        self._counter += 1
        return self._new_aux(
            f"__aux_curl_{operand.field.name}_{self._counter}", inner,
        )

    def get_or_create_aux_for_cross(self, inner: Cross) -> FieldRef:
        """Auxiliary unknown for `inner = Cross(unknown_vec, bg_vec)` (or
        bg-first). The name encodes both operands so the same unknown
        crossed with different backgrounds gets distinct auxiliaries."""
        sig = _expr_signature(inner)
        if sig in self.cache:
            return FieldRef(self.cache[sig])
        assert isinstance(inner.a, FieldRef) and isinstance(inner.b, FieldRef)
        self._counter += 1
        return self._new_aux(
            f"__aux_cross_{inner.a.field.name}_x_{inner.b.field.name}_"
            f"{self._counter}",
            inner,
        )


class _DropTerm:
    """Sentinel returned by _rewrite_expr for a term that is identically
    zero (a composed site whose background vector is zero). Sum handlers
    filter it out."""

    def __repr__(self):
        return "<DROP_TERM>"


DROP_TERM = _DropTerm()


def _rewrite_expr(expr: Expression, ctx: _AugContext) -> Expression:
    """Rewrite `expr`, replacing composed-unknown sites with auxiliary
    FieldRefs (when ctx.prefer_aux_free is False) and dropping
    identically-zero composed terms. Returns `expr` itself when nothing
    changed, DROP_TERM when the whole expression is zero."""
    if isinstance(expr, FieldRef):
        return expr
    if isinstance(expr, Cross):
        a_new = _rewrite_expr(expr.a, ctx)
        b_new = _rewrite_expr(expr.b, ctx)
        if a_new is DROP_TERM or b_new is DROP_TERM:
            return DROP_TERM
        if ctx.prefer_aux_free:
            if (a_new is expr.a) and (b_new is expr.b):
                return expr
            return Cross(a_new, b_new)
        if _is_unknown_curl_of_vec(a_new) and _is_bare_vec_fieldref(
            b_new, background=True
        ):
            if _is_identically_zero_bg_vector(b_new):
                return DROP_TERM
            aux_ref = ctx.get_or_create_aux_for_curl(a_new)
            return Cross(aux_ref, b_new)
        if _is_bare_vec_fieldref(a_new, background=True) and (
            _is_unknown_curl_of_vec(b_new)
        ):
            if _is_identically_zero_bg_vector(a_new):
                return DROP_TERM
            aux_ref = ctx.get_or_create_aux_for_curl(b_new)
            return Cross(a_new, aux_ref)
        if (a_new is expr.a) and (b_new is expr.b):
            return expr
        return Cross(a_new, b_new)
    if isinstance(expr, Sum):
        new_children = tuple(_rewrite_expr(c, ctx) for c in expr.children)
        kept = tuple(c for c in new_children if c is not DROP_TERM)
        if not kept:
            return DROP_TERM
        if all(n is o for n, o in zip(new_children, expr.children)) and (
            len(kept) == len(new_children)
        ):
            return expr
        result = kept[0]
        for c in kept[1:]:
            result = result + c
        return result
    if isinstance(expr, Negation):
        inner = _rewrite_expr(expr.operand, ctx)
        if inner is DROP_TERM:
            return DROP_TERM
        if inner is expr.operand:
            return expr
        return -inner
    if isinstance(expr, ScalarMultiple):
        new_scalar = _rewrite_expr(expr.scalar, ctx)
        new_operand = _rewrite_expr(expr.operand, ctx)
        if new_scalar is DROP_TERM or new_operand is DROP_TERM:
            return DROP_TERM
        if new_scalar is expr.scalar and new_operand is expr.operand:
            return expr
        return ScalarMultiple(new_scalar, new_operand)
    if isinstance(expr, Curl):
        new_inner = _rewrite_expr(expr.operand, ctx)
        if new_inner is DROP_TERM:
            return DROP_TERM
        if ctx.prefer_aux_free:
            if new_inner is expr.operand:
                return expr
            return Curl(new_inner)
        if _is_cross_unknown_bg(new_inner):
            assert isinstance(new_inner, Cross)
            if _is_bare_vec_fieldref(new_inner.a, background=True):
                if _is_identically_zero_bg_vector(new_inner.a):
                    return DROP_TERM
            if _is_bare_vec_fieldref(new_inner.b, background=True):
                if _is_identically_zero_bg_vector(new_inner.b):
                    return DROP_TERM
            aux_ref = ctx.get_or_create_aux_for_cross(new_inner)
            return Curl(aux_ref)
        if new_inner is expr.operand:
            return expr
        return Curl(new_inner)
    if isinstance(expr, Div):
        new_inner = _rewrite_expr(expr.operand, ctx)
        if new_inner is DROP_TERM:
            return DROP_TERM
        if new_inner is expr.operand:
            return expr
        return Div(new_inner)
    if isinstance(expr, Grad):
        new_inner = _rewrite_expr(expr.operand, ctx)
        if new_inner is DROP_TERM:
            return DROP_TERM
        if new_inner is expr.operand:
            return expr
        return Grad(new_inner)
    if isinstance(expr, Lap):
        new_inner = _rewrite_expr(expr.operand, ctx)
        if new_inner is DROP_TERM:
            return DROP_TERM
        if new_inner is expr.operand:
            return expr
        return Lap(new_inner)
    if isinstance(expr, LapH):
        new_inner = _rewrite_expr(expr.operand, ctx)
        if new_inner is DROP_TERM:
            return DROP_TERM
        if new_inner is expr.operand:
            return expr
        return LapH(new_inner)
    if isinstance(expr, Dt):
        new_inner = _rewrite_expr(expr.operand, ctx)
        if new_inner is DROP_TERM:
            return DROP_TERM
        if new_inner is expr.operand:
            return expr
        return Dt(new_inner)
    if isinstance(expr, Dot):
        a_new = _rewrite_expr(expr.a, ctx)
        b_new = _rewrite_expr(expr.b, ctx)
        if a_new is DROP_TERM or b_new is DROP_TERM:
            return DROP_TERM
        if a_new is expr.a and b_new is expr.b:
            return expr
        return Dot(a_new, b_new)
    return expr


# ---------------------------------------------------------------------------
# EquationSystem augmentation
# ---------------------------------------------------------------------------

def augment_equation_system(
    system: EquationSystem,
    prefer_aux_free: bool = True,
) -> EquationSystem:
    """Apply the augmentation rewrite to a system.

    Parameters
    ----------
    system : EquationSystem
    prefer_aux_free : bool, default True
        True leaves ``Cross(Curl(unknown), bg)`` and
        ``Curl(Cross(unknown, bg))`` for the walker's composed
        dispatchers; False unfolds them into auxiliary vector unknowns
        with defining equations.

    Returns the original object when no equation changes, otherwise a
    new EquationSystem with the rewritten equations, the defining
    equations appended, and the same BCs and gauge unknowns. An
    equation whose whole lhs is identically zero raises ValueError.
    """
    ctx = _AugContext(prefer_aux_free=prefer_aux_free)
    new_equations: List[Equation] = []
    changed = False
    for eq in system.equations:
        new_lhs = _rewrite_expr(eq.lhs, ctx)
        if new_lhs is DROP_TERM:
            raise ValueError(
                f"equation {eq.name!r} vanishes identically: every term is a "
                f"composed product with a zero background vector"
            )
        if new_lhs is eq.lhs:
            new_equations.append(eq)
        else:
            changed = True
            new_equations.append(Equation(lhs=new_lhs, name=eq.name))
    if not changed and not ctx.new_equations:
        return system
    all_equations = new_equations + ctx.new_equations
    return EquationSystem(
        equations=all_equations,
        boundary_conditions=list(system.boundary_conditions),
        gauge_unknowns=list(system.gauge_unknowns),
    )
