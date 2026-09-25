"""spdee.ir.tasks — backend-agnostic description of matrix contributions.

The walker (walker.py) turns an EquationSystem into a list of
OperatorTask records. Each task describes one contribution: "this kind
of operator, acting on this unknown, in this equation's row block, with
this background coefficient and sign, going to A, B, or the source
vector b". A backend's compiler maps each task kind to its own sparse
builders; the same walker drives every backend.

Every differential task kind has a natural r-power n: the operator is
emitted multiplied through by r^n so that its coefficients are
polynomial in r near the origin, and the equation's mass becomes
diag(r^n). Pointwise kinds have n = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from spdee.ir.backgrounds import BackgroundComponent


# ===========================================================================
# Task kinds
# ===========================================================================

# Scalar kinds:
TASK_IDENTITY_SCALAR = "identity_scalar"
TASK_LAP_SCALAR = "lap_scalar"
TASK_LAPH_SCALAR = "laph_scalar"      # angular (horizontal) Laplacian
TASK_SOURCE_SCALAR = "source_scalar"  # pure-background scalar term -> b

# Vector kinds:
TASK_IDENTITY_VECTOR = "identity_vector"
TASK_LAP_VECTOR = "lap_vector"
TASK_SOURCE_VECTOR = "source_vector"  # pure-background vector term -> b

TASK_DIV_TENSOR_VECTOR = "div_tensor_vector"
# Div(tensor_expr) in a vector equation. The tensor expression is
# reduced to covariant spin components T_{sigma_a, sigma_b}, linear in
# the placeholders of one vector unknown; the divergence is emitted as
# r^2 (Div T)^{sigma_1} in U = r T variables. r-power 2.

TASK_CURL_VECTOR = "curl_vector"
# Curl(vector_unknown) in a vector equation, via the V19 spin-basis
# curl formula multiplied through by r. r-power 1.

TASK_CROSS_VECTOR = "cross_vector"
# Cross(bg_vec, unknown_vec) or Cross(unknown_vec, bg_vec); pointwise
# coupling of the unknown's spin components through the background's.
# metadata['side'] is 'left' (bg x unknown) or 'right' (unknown x bg).
# r-power 0.

TASK_CROSS_CURL_UNK_BG = "cross_curl_unk_bg"
# Cross(Curl(unknown_vec), bg_vec) or Cross(bg_vec, Curl(unknown_vec)):
# the Lorentz force (curl b) x B0. Assembled as the product of the
# elementary Cross block and the elementary Curl block, so no auxiliary
# unknown is needed. metadata['side'] as for TASK_CROSS_VECTOR.
# r-power 1 (that of Curl).

TASK_CURL_CROSS_UNK_BG = "curl_cross_unk_bg"
# Curl(Cross(unknown_vec, bg_vec)) or Curl(Cross(bg_vec, unknown_vec)):
# the induction term curl(v x B0). Assembled as Curl block times Cross
# block. metadata['inner_side'] gives the inner Cross ordering.
# r-power 1 (that of Curl).

TASK_CURL_CURL_VECTOR = "curl_curl_vector"
# Curl(Curl(unknown_vec)) in a vector equation: the resistive term of an
# induction equation written as -Curl(Curl(b)), or the current of a
# vector potential. r-power 2.

TASK_CROSS_CURL_CURL_UNK_BG = "cross_curl_curl_unk_bg"
# Cross(Curl(Curl(unknown_vec)), bg_vec) or Cross(bg_vec, Curl(Curl(...))):
# the Lorentz force of the vector-potential formulation, j = Curl(Curl(A)).
# metadata['side'] as for TASK_CROSS_VECTOR. r-power 2.

TASK_GRAD_SCALAR = "grad_scalar"
# Grad(scalar_unknown) in a vector equation, via the V19 spin-basis
# gradient multiplied through by r*sqrt(2) (so the sigma = +-1 rows are
# polynomial in r); the equation's mass is diag(r*sqrt(2)). r-power 1.

TASK_DIV_VECTOR_SCALAR = "div_vector_scalar"
# Div(vector_unknown) in a scalar equation, via the V19 spin-basis
# divergence multiplied through by r. r-power 1.

TASK_GRAD_DIV_VECTOR = "grad_div_vector"
# Grad(Div(vector_unknown)) in a vector equation: the bulk-divergence
# piece of the compressible viscous stress. Second-order, multiplied
# through by r^2 (mixed d_r d_theta stencil included). r-power 2.

TASK_DOT_BG_GRAD_VEC = "dot_bg_grad_vec"
# Dot(bg_vec, Grad(unknown_vec)): the advection operator (v0 . grad) v.
# Contracted as (v0 . grad v)^{sigma_3} =
# sum_{sigma_a} v0^{sigma_a} (Grad v)_{sigma_a, -sigma_3}, multiplied
# through by r. r-power 1.

TASK_DOT_BG_TENSOR_VEC = "dot_bg_tensor_vec"
# Dot(bg_tensor, unknown_vec): pointwise matrix-times-vector. The
# background's physical components T_ij are transformed to the matvec
# spin form M = U T U^dagger so that w^{sigma_out} =
# sum_{sigma_in} M^{sigma_out, sigma_in} v^{sigma_in}. r-power 0.

TASK_DOT_BG_VEC_TENSOR_UNK = "dot_bg_vec_tensor_unk"
# Dot(bg_vec, tensor_expr(unknown_vec)) -> vector, e.g. the
# grad(ln rho0) . T term of the compressible viscous force. The tensor
# expression is reduced to covariant components T_{sigma_a, sigma_b}
# in v placeholders and contracted as
# (Dot(a, T))^{sigma_out} = sum_{sigma_in} a^{sigma_in} T_{sigma_in, -sigma_out}.
# metadata['rT_components'] holds r * T_{sigma_a, sigma_b}. r-power 1.

TASK_DOT_BG_VEC_GRAD_SCALAR = "dot_bg_vec_grad_scalar"
# Dot(bg_vec, Grad(unknown_scalar)) -> scalar, e.g. a stratification
# term kappa (d ln P0/dr) d_r s. Built from the r*sqrt(2) Grad block and
# the spin contraction a^0 b^0 + a^+ b^- + a^- b^+. r-power 1.

TASK_DOT_BG_VEC_UNK_VEC_SCALAR = "dot_bg_vec_unk_vec_scalar"
# Dot(bg_vec, unknown_vec) -> scalar, e.g. v . grad(s0) in a linearized
# energy equation. The physical contraction a_r b_r + a_th b_th + a_ph b_ph
# is a^0 b^0 + a^+ b^- + a^- b^+ in spin components. r-power 0.

TASK_BG_VEC_TIMES_UNK_SCALAR = "bg_vec_times_unk_scalar"
# unknown_scalar * bg_vec -> vector, e.g. the buoyancy term -s g. At
# each grid point output^sigma = s * bg^sigma. Block shape (3N, N).
# r-power 0.

TASK_BG_VEC_TIMES_DIV_UNK_VEC = "bg_vec_times_div_unk_vec"
# Div(unknown_vec) * bg_vec -> vector, e.g. the trace piece
# -(2/3)(div v) grad(ln rho0) of the compressible viscous force. At
# each grid point output^sigma = bg^sigma * (div v), built from the
# r-scaled Div block. Block shape (3N, 3N). r-power 1 (that of Div).

# Tensor-unknown kinds (a rank-2 unknown T; the collocated backend has no
# placement for them and refuses these kinds, the staggered general
# assembler places T on the C-grid, docs/staggered_general_design.md 2b):
TASK_IDENTITY_TENSOR = "identity_tensor"
# T in a tensor equation (TARGET_B inside Dt, else TARGET_A). r-power 0.

TASK_TENSOR_OF_VECTOR = "tensor_of_vector"
# A tensor expression linear in one vector unknown (Grad(u),
# Transpose(Grad(u)), sums and background multiples) as a term of a
# tensor equation, e.g. the first-order reduction T - Grad(u) = 0.
# metadata['rT_components'] holds r T_{sigma_a, sigma_b} in u
# placeholders and 'tensor_symmetric' the reducer's verdict. r-power 1.

TASK_DIV_TENSOR_UNK = "div_tensor_unk"
# Div(T) of an unknown tensor in a vector equation. r-power 1.

TASK_DOT_BG_VEC_UNK_TENSOR = "dot_bg_vec_unk_tensor"
# Dot(B0, T) (metadata['side'] 'left') or Dot(T, B0) ('right') of an
# unknown tensor with a background vector, in a vector equation.
# r-power 0.

TASK_SOURCE_TENSOR = "source_tensor"
# pure-background tensor term of a tensor equation -> b. r-power 0.

# Matrix targets:
TARGET_A = "A"      # operator side (acts on unknowns)
TARGET_B = "B"      # eigenvalue side (Dt-marked terms)
TARGET_RHS = "b"    # source vector


# ===========================================================================
# OperatorTask
# ===========================================================================

@dataclass(frozen=True)
class OperatorTask:
    """One backend-agnostic matrix contribution.

    Attributes
    ----------
    kind : str
        Task kind (TASK_*); the backend dispatches on it.
    target : str
        TARGET_A, TARGET_B, or TARGET_RHS.
    equation_name : str
        Equation this task belongs to (selects the row block).
    unknown_name : Optional[str]
        Unknown the operator acts on; None for source terms.
    sign : int
        +1 or -1.
    coefficient : Optional[BackgroundComponent]
        Background scalar prefactor; None means 1. For
        TASK_SOURCE_SCALAR this is the source value itself.
    source_field : optional
        Background VectorField / TensorField referenced by the task
        (the source of TASK_SOURCE_VECTOR, or the background operand of
        Cross, Dot and product tasks). The backend evaluates its
        components on the grid.
    metadata : dict
        Operator-specific data; always contains the azimuthal mode 'm'.
    """
    kind: str
    target: str
    equation_name: str
    unknown_name: Optional[str]
    sign: int = 1
    coefficient: Optional[BackgroundComponent] = None
    source_field: Optional[object] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.sign not in (-1, 1):
            raise ValueError(f"sign must be +/- 1; got {self.sign}")
        if self.target not in (TARGET_A, TARGET_B, TARGET_RHS):
            raise ValueError(
                f"Invalid target {self.target!r}; must be one of "
                f"{(TARGET_A, TARGET_B, TARGET_RHS)}"
            )


def bind_parameters(tasks, params):
    """Substitute parameter values into a list of OperatorTasks.

    A system may carry free SymPy symbols for physical parameters
    (``Ek * Lap(v)`` with ``Ek = sympy.Symbol('Ek')``). The walk is then
    independent of the parameter values, and this function specializes
    a walked task list to concrete values, so a parameter sweep walks
    once per azimuthal mode and binds once per parameter value.

    Substitution covers a task's SymPy-backed ``coefficient`` and any
    SymPy expressions and SymPy-backed components nested in
    ``metadata``. Parameters baked into a numeric background component
    are not touched. Symbols are matched by name, so a key ``"Ek"`` or
    ``Symbol("Ek")`` binds ``Symbol("Ek", positive=True)`` too; keys that
    do not appear are ignored. A symbol other than the coordinates r and
    theta left unbound in any task raises ValueError: the backends
    cannot evaluate it.

    Parameters
    ----------
    tasks : list[OperatorTask]
    params : dict {sympy.Symbol or str: value}

    Returns
    -------
    list[OperatorTask]
        Tasks with parameters substituted; unchanged tasks are reused.
    """
    import dataclasses
    from spdee.ir.backgrounds import SymPyComponent, as_background_component
    binder = _ParameterBinder(params)
    out = []
    for t in tasks:
        new_coef = t.coefficient
        if isinstance(t.coefficient, SymPyComponent):
            new_coef = as_background_component(binder.expr(t.coefficient.expr))
        new_meta = binder.nested(t.metadata) if t.metadata else t.metadata
        if binder.unbound:
            raise ValueError(
                f"unbound symbols {binder.unbound_names()} in the {t.kind!r} term on "
                f"{t.unknown_name!r} of equation {t.equation_name!r}; pass "
                f"their values in params"
            )
        if new_coef == t.coefficient and new_meta == t.metadata:
            out.append(t)
        else:
            out.append(dataclasses.replace(
                t, coefficient=new_coef, metadata=new_meta,
            ))
    return out


class _ParameterBinder:
    """Substitutes parameter values, matched by symbol name, into SymPy
    expressions and the SymPy-backed components, dicts, lists, tuples
    and BCTasks nested in a task's payload; collects the symbols other
    than the coordinates left unbound."""

    def __init__(self, params):
        from spdee.ir.fields import r as _r, theta as _theta
        self.by_name = {str(k): v for k, v in params.items()}
        self.coordinates = {_r, _theta}
        self.unbound = []

    def unbound_names(self):
        return sorted({s.name for s in self.unbound})

    def expr(self, expr):
        subs_map = {s: self.by_name[s.name] for s in expr.free_symbols
                    if s.name in self.by_name}
        if subs_map:
            expr = expr.subs(subs_map)
        free = expr.free_symbols - self.coordinates
        if free:
            self.unbound.extend(free)
        return expr

    def nested(self, obj):
        import dataclasses
        import sympy as _sp
        from spdee.ir.backgrounds import SymPyComponent, as_background_component
        if isinstance(obj, _sp.Basic):
            return self.expr(obj)
        if isinstance(obj, SymPyComponent):
            return as_background_component(self.expr(obj.expr))
        if isinstance(obj, dict):
            return {k: self.nested(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.nested(v) for v in obj)
        if isinstance(obj, BCTask):
            value = None if obj.value is None else self.nested(obj.value)
            metadata = self.nested(obj.metadata) if obj.metadata else obj.metadata
            if value == obj.value and metadata == obj.metadata:
                return obj
            return dataclasses.replace(obj, value=value, metadata=metadata)
        return obj


# ===========================================================================
# BC kinds
# ===========================================================================

BC_DIRICHLET_SCALAR = "dirichlet_scalar"
BC_DIRICHLET_VECTOR_PROJECTION = "dirichlet_vector_projection"
BC_SCALAR_LINEAR_FROM_TENSOR = "scalar_linear_from_tensor"
BC_NEUMANN_SCALAR_RADIAL = "neumann_scalar_radial"
BC_PROJECTION_CURL_CROSS = "projection_curl_cross"
BC_LINEAR_COMBINATION = "linear_combination"

# The single-shape kinds a BC_LINEAR_COMBINATION term may take.
BC_TERM_KINDS = (
    BC_DIRICHLET_SCALAR, BC_DIRICHLET_VECTOR_PROJECTION,
    BC_SCALAR_LINEAR_FROM_TENSOR, BC_NEUMANN_SCALAR_RADIAL,
    BC_PROJECTION_CURL_CROSS,
)

# metadata['tangent'] of a BC_LINEAR_COMBINATION: the unit tag shared by
# its tangential projections, BC_TANGENT_MIXED when both occur, None
# when the row has none.
BC_TANGENT_MIXED = "mixed"

# Unit-vector tags for vector-projection BCs.
BC_UNIT_R_HAT = "r_hat"
BC_UNIT_THETA_HAT = "theta_hat"
BC_UNIT_PHI_HAT = "phi_hat"

# Boundary locations (same strings as Boundary in equations.py; kept
# here so this module does not import equations.py).
BC_LOC_R_INNER = "r=0"
BC_LOC_R_OUTER = "r=1"


@dataclass(frozen=True)
class BCTask:
    """One backend-agnostic boundary-condition record.

    Kinds:

    BC_DIRICHLET_SCALAR
        p(loc) = value for a scalar unknown p.

    BC_DIRICHLET_VECTOR_PROJECTION
        (unit . v)(loc) = value for a vector unknown v, with `unit` one
        of the BC_UNIT_* tags: r_hat.v = v^0, theta_hat.v =
        (v^+ + v^-)/sqrt(2), phi_hat.v = i(v^- - v^+)/sqrt(2). No-slip
        is three of these, one per unit vector.

    BC_SCALAR_LINEAR_FROM_TENSOR
        A scalar constraint from a tensor projection
        Dot(u, Dot(T_expr, w)) = value: a linear combination of one
        vector unknown's spin components and their r/theta derivatives.
        metadata keys:
          'scalar_expr'  : sp.Expr in the placeholders
          'placeholders' : {unknown_name: {sigma: sp.Function}}
          'host_unknown' : str, the unknown whose row hosts the constraint
          'host_sigma'   : int, the spin block (-1, 0, +1) hosting it

    BC_NEUMANN_SCALAR_RADIAL
        (r_hat . grad s)(loc) = d_r s|_loc = value for a scalar unknown
        s; discretized with a one-sided radial stencil.

    BC_PROJECTION_CURL_CROSS
        (unit . X)(loc) = value with X a linear combination, with
        constant coefficients, of Curl(u) and Cross(u, B0) / Cross(B0, u)
        terms of unknown vectors u (the tangential electric field
        eta Curl(b) - Cross(v, B0) of a conducting wall). `unit` is the
        BC_UNIT_* tag; `unknown_name` the unknown under the Curl (else the
        first unknown); metadata['terms'] lists the terms as dicts with
        'kind' ('curl' / 'cross'), 'coefficient' (SymPy), 'unknown',
        'background' (the background VectorField of a cross, else None)
        and 'side' ('right' for Cross(u, B0), 'left' for Cross(B0, u)).

    BC_LINEAR_COMBINATION
        sum_i c_i X_i(loc) = value, one row combining several of the
        shapes above on one or more unknowns: the Robin / Biot wall
        Dot(r_hat, Grad(p)) + h p = 0, Navier slip Dot(t_hat, v)
        + l_s Dot(t_hat, Dot(strain, r_hat)) = 0, the normal stress
        Dot(r_hat, Dot(strain, r_hat)) - c p = 0. The walker emits it
        for a row with several unknown terms (and for one term under a
        numeric coefficient beside a value); a single-shape row keeps
        its own kind above, its value divided by the coefficient.
        metadata keys:
          'terms'       : list of (coefficient, term) pairs. `term` is a
                          homogeneous BCTask (value None) of one of the
                          BC_TERM_KINDS at the same location, and
                          `coefficient` a BackgroundComponent: a number
                          or SymPy expression in r, theta and the
                          parameters (SymPyComponent), or a numeric
                          background (NumericComponent, or a Composite
                          when multiplied). The sign of the term is in
                          the coefficient.
          'unknowns'    : tuple of every unknown the row touches, in
                          first-seen order.
          'host_unknown': the unknown whose rows host the row: the
                          highest-rank unknown touched (first seen on a
                          tie); `unknown_name` repeats it.
          'host_sigma'  : the spin block hosting it on a vector host,
                          +1 for a theta_hat row, -1 for a phi_hat row,
                          0 otherwise; None on a scalar host.
          'tangent'     : BC_UNIT_THETA_HAT / BC_UNIT_PHI_HAT when every
                          tangential projection of the row (a Dirichlet
                          projection, a curl-cross projection, the outer
                          unit of a tensor projection) is on that unit,
                          BC_TANGENT_MIXED when both occur, None when the
                          row has no tangential projection.
          'pairing_key' : the structure of the boundary expression with
                          theta_hat and phi_hat made alike, keyed with
                          the location (`boundary_pairing_key` in
                          walker.py); the rows of a theta / phi pair on
                          one expression share it.

        Consumer contract. A core assembles the row at the wall
        r = r_loc as sum_i c_i(r_loc, theta) * row_i = value(r_loc,
        theta), where row_i is the row the core assembles for the term
        alone (its own homogeneous kind): the wall value of a scalar,
        the unit projection of a vector, the one-sided radial
        derivative of a scalar, the scalar_expr of a tensor projection
        with its derivatives, the curl / cross combination projected on
        its unit. A coefficient is evaluated at the wall; a spectral
        core that keeps one row per harmonic takes a radial profile or
        a number there and refuses a theta-dependent coefficient, an FD
        core evaluates it at every wall node. The row is homogeneous
        when `value` is None. The terms of a single row never carry a
        value of their own; a background part of a tensor projection
        has already moved into the row's value, scaled by the term's
        coefficient.

        Pairing rule. A theta_hat row and a phi_hat row written on the
        same expression (the same pairing_key at the same location) are
        the two components of one tangential vector condition: a
        spectral core with vector harmonics (Dedalus, sh_fd) imposes
        them as one angular row and refuses a lone tangential row, an
        FD core imposes them as two scalar rows, the theta_hat one on
        the sigma = +1 block and the phi_hat one on sigma = -1 of the
        host. A row with BC_TANGENT_MIXED tangent (theta_hat and
        phi_hat projections in one expression) is a scalar row on an FD
        core and is refused by the spectral cores. The single-shape
        kinds pair the same way; `boundary_pairing_key` gives their key
        from the system's BoundaryCondition.

    Attributes
    ----------
    kind : str
    location : str
        BC_LOC_R_INNER or BC_LOC_R_OUTER (radial boundaries only).
    unknown_name : str
        The unknown being constrained.
    value : Optional[BackgroundComponent]
        Boundary value as a background scalar; None means homogeneous.
    name : str
        Label for diagnostics.
    unit : Optional[str]
        BC_UNIT_* tag for BC_DIRICHLET_VECTOR_PROJECTION and
        BC_PROJECTION_CURL_CROSS; None otherwise.
    metadata : dict
        See BC_SCALAR_LINEAR_FROM_TENSOR, BC_PROJECTION_CURL_CROSS and
        BC_LINEAR_COMBINATION; empty for the other kinds.
    """
    kind: str
    location: str
    unknown_name: str
    value: Optional[BackgroundComponent] = None
    name: str = ""
    unit: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in BC_TERM_KINDS + (BC_LINEAR_COMBINATION,):
            raise ValueError(
                f"Unknown BC kind {self.kind!r}; supported: "
                f"{BC_DIRICHLET_SCALAR!r}, {BC_DIRICHLET_VECTOR_PROJECTION!r}, "
                f"{BC_SCALAR_LINEAR_FROM_TENSOR!r}, "
                f"{BC_NEUMANN_SCALAR_RADIAL!r}, {BC_PROJECTION_CURL_CROSS!r}, "
                f"{BC_LINEAR_COMBINATION!r}"
            )
        if self.location not in (BC_LOC_R_INNER, BC_LOC_R_OUTER):
            raise ValueError(
                f"Unsupported BC location {self.location!r}; only radial "
                f"boundaries (r=inner, r=outer) are supported"
            )
        if self.kind == BC_LINEAR_COMBINATION:
            self._check_linear_combination()

    def _check_linear_combination(self):
        terms = self.metadata.get("terms")
        if not terms:
            raise ValueError(
                f"BC {self.name!r}: a linear combination needs at least one term"
            )
        for coefficient, term in terms:
            if not isinstance(coefficient, BackgroundComponent):
                raise ValueError(
                    f"BC {self.name!r}: term coefficient must be a "
                    f"BackgroundComponent; got {type(coefficient).__name__}"
                )
            if not isinstance(term, BCTask) or term.kind not in BC_TERM_KINDS:
                raise ValueError(
                    f"BC {self.name!r}: a linear-combination term must be a "
                    f"BCTask of one of {BC_TERM_KINDS}; got {term!r}"
                )
            if term.location != self.location or term.value is not None:
                raise ValueError(
                    f"BC {self.name!r}: term {term.name!r} must be homogeneous "
                    f"at {self.location!r}"
                )
        for key in ("unknowns", "host_unknown", "host_sigma", "tangent", "pairing_key"):
            if key not in self.metadata:
                raise ValueError(
                    f"BC {self.name!r}: linear combination metadata lacks {key!r}"
                )
        if self.metadata["host_unknown"] != self.unknown_name:
            raise ValueError(
                f"BC {self.name!r}: unknown_name {self.unknown_name!r} is not the "
                f"host {self.metadata['host_unknown']!r}"
            )


def bind_bc_parameters(bc_tasks, params):
    """`bind_parameters` for BCTasks: substitutes parameter values into a
    task's SymPy-backed value and into the SymPy expressions, components
    and nested term tasks of its metadata (the coefficients and terms of
    a BC_LINEAR_COMBINATION, the curl-cross coefficients, the tensor
    scalar_expr). Placeholders are functions of r and theta and stay.
    A symbol other than r and theta left unbound raises ValueError.
    Unchanged tasks are reused."""
    binder = _ParameterBinder(params)
    out = []
    for t in bc_tasks:
        bound = binder.nested(t)
        if binder.unbound:
            raise ValueError(
                f"unbound symbols {binder.unbound_names()} in boundary condition "
                f"{t.name!r} on {t.unknown_name!r}; pass their values in params"
            )
        out.append(bound)
    return out
