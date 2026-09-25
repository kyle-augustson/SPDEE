"""spdee.ir.constraints — backend-agnostic classification of a system's
equations from its walked tasks (docs/ir_boundary_and_constraints.md,
docs/staggered_general_design.md section 4).

Every equation of an EquationSystem is one of

    evolution     a TARGET_B task on one unknown u: the rows live on u's
                  DOFs and the mass is the identity there. A system with
                  no Dt anywhere is the eigenvalue operator itself,
                  L x = lambda x: a vector equation whose only vector
                  unknown is u is read as u's evolution equation with a
                  unit Dt(u) ('implicit_mass', the Dt-less rule).
    saddle_point  Div(M u) = 0, M a scalar background or 1, with a scalar
                  unknown p entering u's evolution equation through
                  Grad(p) alone: p is the multiplier, the rows are p's,
                  the mass is zero there. At m = 0 the constant p is a
                  null mode and the record names the gauge, 'integral'
                  when the system declares p a gauge unknown, else
                  'point' (one row replaced by p = 0).
    involution    Div(b) = 0 with no multiplier for a vector b whose
                  evolution is a curl of the state, Curl(Cross(v, B0)),
                  plus its own diffusion Lap(b) or Curl(Curl(b)): the
                  evolution preserves the constraint. The reading is
                  also implied, without a written constraint, for such
                  an evolution (`add_magnetic_field` writes none).
    definition    T - tensor(u) = 0, a tensor equation without Dt whose
                  terms are one unknown tensor and tensor expressions of
                  one vector unknown; T is eliminated.

Anything else is refused with NotImplementedError quoting the equation,
the shape found and the supported shapes.

The `mimetic` flag says what the consuming core does with an involution.
A mimetic core (the C-grid) has Div Curl = 0 exactly, drops the written
constraint row and keeps b on the face fluxes
(`involution_treatment = "dropped"`). A non-mimetic core (collocated,
spin-weighted harmonics, Dedalus) cannot drop it: the record then
prescribes the divergence-cleaning projection
(`involution_treatment = "saddle_point"`): an auto-introduced scalar
multiplier `__div_<b>` enters b's evolution equation through + Grad of
it, the constraint Div(b) = 0 becomes a saddle point on the multiplier's
rows (the core adds the row itself when the constraint was implied),
with the mean-zero gauge at m = 0 and Dirichlet-zero walls on the
multiplier (`divergence_cleaning`). The classification of the equations
is the same under both flags; only the treatment differs.

The placement of every unknown is recorded twice: `rank` (0, 1, 2) for
the collocated and spectral cores and `placement` ('center', 'face',
'tensor' for the six symmetric strain components, 'tensor9' for all
nine) for the C-grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sympy as sp

from spdee.ir.tasks import (
    OperatorTask, TARGET_A, TARGET_B,
    TASK_CURL_CROSS_UNK_BG, TASK_CURL_CURL_VECTOR, TASK_CURL_VECTOR,
    TASK_DIV_VECTOR_SCALAR, TASK_DOT_BG_VEC_UNK_VEC_SCALAR, TASK_GRAD_SCALAR,
    TASK_IDENTITY_TENSOR, TASK_LAP_VECTOR, TASK_TENSOR_OF_VECTOR,
    BC_LOC_R_INNER, BC_LOC_R_OUTER,
)

__all__ = [
    "ClassificationRecord", "DivergenceCleaning", "classify",
    "SUPPORTED_SHAPES", "CURL_KINDS", "GEOMETRIES", "INVOLUTION_DROPPED",
    "INVOLUTION_SADDLE_POINT", "describe_tasks", "div_shape", "tensor_terms",
    "auto_multiplier_name",
]

GEOMETRIES = ("shell", "ball")

INVOLUTION_DROPPED = "dropped"
INVOLUTION_SADDLE_POINT = "saddle_point"

# the curl evolutions of an involution
CURL_KINDS = (TASK_CURL_CROSS_UNK_BG, TASK_CURL_VECTOR, TASK_CURL_CURL_VECTOR)

SUPPORTED_SHAPES = (
    "(1) Div(M u) = 0 with a scalar unknown p entering u's evolution "
    "equation through Grad(p) alone (saddle point, p the multiplier), "
    "(2) Div(b) = 0 with Dt(b) = Curl(...) [+ eta Lap(b) or "
    "- eta Curl(Curl(b))] and no multiplier (involution, constrained "
    "transport) and (3) T - tensor(u) = 0 defining an unknown tensor T "
    "(definition)"
)

_RANK = {"scalar": 0, "vector": 1, "tensor": 2}


def auto_multiplier_name(vector: str) -> str:
    """The scalar multiplier a non-mimetic core introduces for the
    involution of `vector`."""
    return f"__div_{vector}"


@dataclass(frozen=True)
class DivergenceCleaning:
    """The saddle-point treatment of one involution on a non-mimetic
    core: the scalar `multiplier` (placement 'center', rank 0) enters
    `evolution` (the curl evolution of `vector`) through + Grad of it;
    `constraint` names the written Div(vector) = 0 equation that hosts
    the multiplier's rows, or is None when the constraint was implied
    and the core adds the row Div(vector) = 0 itself.  The multiplier's
    constant must be fixed exactly once: a core that can impose rows on
    the multiplier takes `walls` (Dirichlet zero at those locations) and
    must not also impose `gauge`; a tau core with no wall-row layout for
    the multiplier takes `gauge` ('integral', mean zero, at m = 0) alone.
    Imposing both admits a spurious l = 0 mode of the vector."""
    vector: str
    multiplier: str
    evolution: str
    constraint: Optional[str]
    gauge: Optional[str]
    walls: tuple
    placement: str = "center"
    rank: int = 0


@dataclass
class ClassificationRecord:
    """The classification of one system at one azimuthal mode.

    `equations`, `principal`, `multipliers`, `involutions`, `constraints`
    and `definitions` are the fields of the C-grid classifier
    (`staggered_general.classify_constraints`), with the same keys:

        placement   : {unknown: 'center' | 'face' | 'tensor' | 'tensor9'}
        equations   : {equation: {'class': ..., ...}}; an evolution
                      equation names its 'unknown' (and 'implicit_mass'
                      under the Dt-less rule); a saddle point its
                      'vector', 'multiplier', 'weight', 'div_sign',
                      'div_outer', the multiplier's 'grad_tasks' and its
                      'gauge'; an involution its 'vector', 'evolution'
                      and whether its row is 'dropped'; a definition its
                      'tensor', 'identity' task, 'terms' and 'vector'
        principal   : {equation: unknown whose DOFs carry its rows}
        multipliers : {multiplier: constraint equation}
        involutions : {vector: its curl evolution equation}
        constraints : {vector: 'saddle_point' | 'involution'}
        definitions : {tensor: equation}

    plus

        geometry, mimetic, m
        dt_less               : the Dt-less rule applied
        rank                  : {unknown: 0 | 1 | 2}
        gauges                : {multiplier: 'integral' | 'point' | None}
        involution_treatment  : INVOLUTION_DROPPED (mimetic) or
                                INVOLUTION_SADDLE_POINT (non-mimetic)
        divergence_cleaning   : {vector: DivergenceCleaning} on a
                                non-mimetic core, empty otherwise
    """
    geometry: str
    mimetic: bool
    m: int
    dt_less: bool = False
    rank: Dict[str, int] = field(default_factory=dict)
    placement: Dict[str, str] = field(default_factory=dict)
    equations: Dict[str, dict] = field(default_factory=dict)
    principal: Dict[str, str] = field(default_factory=dict)
    multipliers: Dict[str, str] = field(default_factory=dict)
    involutions: Dict[str, str] = field(default_factory=dict)
    constraints: Dict[str, str] = field(default_factory=dict)
    definitions: Dict[str, str] = field(default_factory=dict)
    gauges: Dict[str, Optional[str]] = field(default_factory=dict)
    involution_treatment: str = INVOLUTION_DROPPED
    divergence_cleaning: Dict[str, DivergenceCleaning] = field(default_factory=dict)

    def staggered_view(self) -> dict:
        """The dict `staggered_general.classify_constraints` returns."""
        return {
            "placement": self.placement, "equations": self.equations,
            "principal": self.principal, "multipliers": self.multipliers,
            "involutions": self.involutions, "constraints": self.constraints,
            "definitions": self.definitions,
        }


def describe_tasks(tasks: List[OperatorTask]) -> str:
    parts = []
    for t in tasks:
        tag = t.kind
        if t.target == TARGET_B:
            tag = f"Dt:{tag}"
        parts.append(f"{tag}({t.unknown_name})" if t.unknown_name else tag)
    return " + ".join(parts) if parts else "no terms"


def div_shape(tasks: List[OperatorTask]):
    """(unknown, weight, outer coefficient, sign) when the task list is
    exactly Div(u) or the tagged pair of Div(M u); None otherwise."""
    divs = [t for t in tasks if t.kind == TASK_DIV_VECTOR_SCALAR]
    if len(divs) != 1:
        return None
    d = divs[0]
    if d.metadata.get("div_weight_role") == "div":
        weight = d.metadata["div_weight"]
        partners = [t for t in tasks if t is not d]
        if len(partners) != 1:
            return None
        g = partners[0]
        if (g.kind != TASK_DOT_BG_VEC_UNK_VEC_SCALAR
                or g.metadata.get("div_weight_role") != "grad"
                or g.metadata.get("div_weight") is not weight
                or g.unknown_name != d.unknown_name or g.sign != d.sign):
            return None
        return d.unknown_name, weight, d.metadata.get("div_outer"), d.sign
    if len(tasks) != 1:
        return None
    return d.unknown_name, None, d.coefficient, d.sign


def tensor_terms(task: OperatorTask, core: str = "constraints"):
    """(a, b) of a tensor task whose expression is a Grad(u) /
    Transpose(Grad(u)) combination with constant weights, from the
    walker's exact record; refused otherwise."""
    weights = task.metadata.get("grad_transpose_weights")
    if weights is not None:
        vals = []
        for w in weights:
            w = sp.sympify(w)
            if w.free_symbols:
                weights = None
                break
            vals.append(complex(w))
    if weights is None:
        raise NotImplementedError(
            f"{core}: the tensor expression of the {task.kind!r} term on "
            f"{task.unknown_name!r} in {task.equation_name!r} is not a constant "
            f"combination a Grad(u) + b Transpose(Grad(u)); background tensors "
            f"times u and background scalars inside the tensor are refused "
            f"(write the scalar outside the tensor expression)")
    return vals[0], vals[1]


def _promote_eigenvalue_equations(by_eq, algebraic, placement,
                                  evolution_eq_of, record) -> None:
    """The Dt-less rule: a vector equation whose only face unknown is u
    is read as u's evolution equation with a unit Dt(u), so the mass
    carries the identity on u's DOFs ('implicit_mass')."""
    for eq in algebraic:
        if eq.kind != "vector":
            continue
        faces = {t.unknown_name for t in by_eq[eq.name]
                 if placement.get(t.unknown_name) == "face"}
        if len(faces) != 1:
            continue
        (u,) = faces
        if u in evolution_eq_of:
            continue
        evolution_eq_of[u] = eq.name
        record.equations[eq.name] = {"class": "evolution", "unknown": u,
                                     "implicit_mass": True}
        record.principal[eq.name] = u


def classify(system, tasks: List[OperatorTask], *, geometry: str, mimetic: bool,
             core: str = "constraints") -> ClassificationRecord:
    """Classify every equation of `system` from its walked `tasks` (the
    module docstring). `geometry` is 'shell' or 'ball' (the walls a
    divergence-cleaning multiplier is pinned on); `mimetic` selects the
    involution treatment; `core` prefixes the refusal messages.

    A constraint that admits both the saddle-point and the involution
    reading is refused naming both; so is an involution whose evolution
    carries terms other than curls of the state and its own diffusion, a
    weighted Div(M b) = 0 without a multiplier, a tensor unknown with no
    tensor equation, and an unknown that is neither under Dt nor a
    multiplier nor a defined tensor.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}; got {geometry!r}")
    unknowns = system.unknowns
    m = int(tasks[0].metadata.get("m", 0)) if tasks else 0
    record = ClassificationRecord(geometry=geometry, mimetic=bool(mimetic), m=m)
    placement = record.placement
    for name, fld in unknowns.items():
        record.rank[name] = _RANK[fld.kind]
        placement[name] = {"scalar": "center", "vector": "face"}.get(fld.kind, "tensor")
    by_eq: Dict[str, List[OperatorTask]] = {eq.name: [] for eq in system.equations}
    for t in tasks:
        by_eq.setdefault(t.equation_name, []).append(t)
    evolution_eq_of: Dict[str, str] = {}
    algebraic = []
    for eq in system.equations:
        ts = by_eq[eq.name]
        under_dt = {t.unknown_name for t in ts if t.target == TARGET_B}
        if len(under_dt) > 1:
            raise NotImplementedError(
                f"{core}: equation {eq.name!r} carries Dt on several "
                f"unknowns {sorted(under_dt)}; one unknown per Dt equation")
        if under_dt:
            (u,) = under_dt
            if u in evolution_eq_of:
                raise NotImplementedError(
                    f"{core}: unknown {u!r} is under Dt in equations "
                    f"{evolution_eq_of[u]!r} and {eq.name!r}")
            evolution_eq_of[u] = eq.name
            record.equations[eq.name] = {"class": "evolution", "unknown": u}
            record.principal[eq.name] = u
        else:
            algebraic.append(eq)
    if not evolution_eq_of:
        record.dt_less = True
        _promote_eigenvalue_equations(by_eq, algebraic, placement,
                                      evolution_eq_of, record)
        algebraic = [eq for eq in algebraic if eq.name not in record.equations]
    appearances: Dict[str, List[OperatorTask]] = {u: [] for u in unknowns}
    for t in tasks:
        if t.unknown_name is not None:
            appearances[t.unknown_name].append(t)

    def refuse(eq, why):
        raise NotImplementedError(
            f"{core}: equation {eq.name!r} is algebraic (no Dt) and "
            f"unrecognised: found {describe_tasks(by_eq[eq.name])}; {why}. The "
            f"assembler pairs only {SUPPORTED_SHAPES}.")

    # tensor definitions: T - tensor(u) = 0 in a tensor equation without Dt
    remaining = []
    for eq in algebraic:
        ts = by_eq[eq.name]
        ids = [t for t in ts if t.kind == TASK_IDENTITY_TENSOR]
        terms = [t for t in ts if t.kind == TASK_TENSOR_OF_VECTOR]
        if eq.kind != "tensor":
            remaining.append(eq)
            continue
        if len(ids) != 1 or len(terms) + 1 != len(ts):
            refuse(eq, "a tensor definition is one unknown tensor plus tensor "
                       "terms of one vector unknown")
        T = ids[0].unknown_name
        if T in evolution_eq_of or T in record.definitions:
            refuse(eq, f"tensor {T!r} already has an equation")
        vectors = {t.unknown_name for t in terms}
        if len(vectors) > 1:
            refuse(eq, f"the definition of {T!r} mixes the vectors {sorted(vectors)}")
        ab = [tensor_terms(t, core) for t in terms]
        record.equations[eq.name] = {
            "class": "definition", "tensor": T, "identity": ids[0],
            "terms": list(zip(terms, ab)),
            "vector": vectors.pop() if vectors else None,
        }
        record.principal[eq.name] = T
        record.definitions[T] = eq.name
    algebraic = remaining
    for name, fld in unknowns.items():
        if fld.kind != "tensor":
            continue
        eq_name = evolution_eq_of.get(name) or record.definitions.get(name)
        if eq_name is None:
            raise NotImplementedError(
                f"{core}: tensor unknown {name!r} has no tensor equation "
                f"(a definition T - tensor(u) = 0 or an evolution equation "
                f"under Dt) whose rows live on its components (docs/"
                f"staggered_general_design.md section 2b)")
        terms = [t for t in by_eq[eq_name] if t.kind == TASK_TENSOR_OF_VECTOR]
        symmetric = all(a == b for a, b in (tensor_terms(t, core) for t in terms))
        placement[name] = "tensor" if symmetric else "tensor9"

    shapes = {eq.name: div_shape(by_eq[eq.name]) for eq in algebraic}
    for eq in algebraic:
        if shapes[eq.name] is None:
            refuse(eq, "the equation is not Div(M u) = 0 of one vector unknown")
    for eq in algebraic:
        u, weight, outer, sign = shapes[eq.name]
        if placement[u] != "face":
            refuse(eq, f"unknown {u!r} has placement {placement[u]!r}, not 'face'")
        u_eq = evolution_eq_of.get(u)
        if u_eq is None:
            refuse(eq, f"vector {u!r} has no evolution equation")
        cands = []
        for p, fld in unknowns.items():
            if fld.kind != "scalar" or p in evolution_eq_of or p in record.multipliers:
                continue
            apps = appearances[p]
            if apps and all(t.kind == TASK_GRAD_SCALAR and t.equation_name == u_eq
                            for t in apps):
                cands.append(p)
        if len(cands) > 1:
            refuse(eq, f"several scalar unknowns {cands} enter {u_eq!r} through Grad "
                       f"alone; the multiplier of Div({u}) is ambiguous")
        a_tasks = [t for t in by_eq[u_eq] if t.target == TARGET_A]
        diffusion = [t for t in a_tasks if t.unknown_name == u
                     and t.kind in (TASK_LAP_VECTOR, TASK_CURL_CURL_VECTOR)]
        curls = [t for t in a_tasks if t.kind in CURL_KINDS and t not in diffusion]
        if cands and curls:
            refuse(eq, f"both readings apply: {cands[0]!r} enters {u_eq!r} through "
                       f"Grad alone (saddle point, {cands[0]} the multiplier of "
                       f"Div({u})) and {u_eq!r} carries the curl terms "
                       f"{describe_tasks(curls)} (involution, Div Curl = 0 with no "
                       f"multiplier); drop Grad({cands[0]}) for constrained "
                       f"transport or write the evolution without curls for the "
                       f"vector-potential saddle point")
        if cands:
            p = cands[0]
            gauge = None
            if m == 0:
                gauge = "integral" if p in system.gauge_unknowns else "point"
            record.equations[eq.name] = {
                "class": "saddle_point", "vector": u, "multiplier": p,
                "weight": weight, "div_sign": sign, "div_outer": outer,
                "grad_tasks": list(appearances[p]), "gauge": gauge,
            }
            record.principal[eq.name] = p
            record.multipliers[p] = eq.name
            record.constraints[u] = "saddle_point"
            record.gauges[p] = gauge
            continue
        if not curls and not diffusion:
            refuse(eq, f"no scalar unknown enters {u_eq!r} through Grad alone and "
                       f"{u_eq!r} has no Curl evolution")
        if weight is not None:
            refuse(eq, f"Div(M {u}) = 0 with a weight is not preserved by the curl "
                       f"evolution {u_eq!r} (Div Curl = 0 holds for the unweighted "
                       f"divergence) and has no multiplier")
        others = [t for t in a_tasks if t.kind != TASK_CURL_CROSS_UNK_BG
                  and not (t.unknown_name == u and t.kind in (
                      TASK_LAP_VECTOR, TASK_CURL_CURL_VECTOR))]
        if others:
            refuse(eq, f"the terms {describe_tasks(others)} of {u_eq!r} are not curls "
                       f"of the state, so they do not preserve Div({u}) = 0; the "
                       f"involution takes Curl(Cross(v, B0)), Lap({u}) and "
                       f"Curl(Curl({u})) only")
        record.equations[eq.name] = {"class": "involution", "vector": u,
                                     "evolution": u_eq, "dropped": bool(mimetic)}
        record.involutions[u] = u_eq
        record.constraints[u] = "involution"
    for u, u_eq in evolution_eq_of.items():
        if placement[u] != "face" or u in record.constraints:
            continue
        a_tasks = [t for t in by_eq[u_eq] if t.target == TARGET_A]
        curls = [t for t in a_tasks if t.kind in CURL_KINDS
                 and not (t.unknown_name == u and t.kind == TASK_CURL_CURL_VECTOR)]
        others = [t for t in a_tasks if t.kind != TASK_CURL_CROSS_UNK_BG
                  and not (t.unknown_name == u and t.kind in (
                      TASK_LAP_VECTOR, TASK_CURL_CURL_VECTOR))]
        if curls and not others:
            record.involutions[u] = u_eq
            record.constraints[u] = "involution"
    placed = set(record.principal.values()) | set(record.involutions)
    leftover = sorted(u for u in unknowns if u not in placed)
    if leftover:
        raise NotImplementedError(
            f"{core}: unknowns {leftover} are neither under Dt nor a "
            f"multiplier of a constraint nor a defined tensor; every unknown "
            f"needs one equation whose rows live on its DOFs")
    if not mimetic:
        _prescribe_divergence_cleaning(system, record, core)
    return record


def _prescribe_divergence_cleaning(system, record: ClassificationRecord, core: str) -> None:
    """The saddle-point treatment of every involution for a non-mimetic
    core (the module docstring)."""
    record.involution_treatment = INVOLUTION_SADDLE_POINT
    written = {info["vector"]: name for name, info in record.equations.items()
               if info["class"] == "involution"}
    walls = (BC_LOC_R_OUTER,) if record.geometry == "ball" else (
        BC_LOC_R_INNER, BC_LOC_R_OUTER)
    for b, u_eq in record.involutions.items():
        multiplier = auto_multiplier_name(b)
        if multiplier in system.unknowns:
            raise NotImplementedError(
                f"{core}: the system already has an unknown named {multiplier!r}, "
                f"the multiplier of the divergence cleaning of {b!r}")
        gauge = "integral" if record.m == 0 else None
        record.divergence_cleaning[b] = DivergenceCleaning(
            vector=b, multiplier=multiplier, evolution=u_eq,
            constraint=written.get(b), gauge=gauge, walls=walls,
        )
        record.gauges[multiplier] = gauge
