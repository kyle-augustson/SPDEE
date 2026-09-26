"""Arakawa-C (staggered) grids, layout and mimetic block builders.

Selected by ``compile_fd2(..., discretization=STAGGERED)``. The pencil
A x = λ B x is put together by ``staggered_general``, which classifies the
walker's tasks and dispatches each to one of the block builders here (and
to the constrained-transport builders of ``staggered_mhd``);
``compile_fd2_staggered`` is that call. This module owns the Gilman-type
staggered layout:

  * scalars (p, ρ₁, s₁) and v_φ at cell centers (r_c, θ_c);
  * v_r on r-edges (r_e, θ_c); v_θ on θ-edges (r_c, θ_e);
  * staggered gradients in r and θ (2-point by default, or a centered
    Lagrange stencil of any even order);
  * divergence = norm-adjoint of the gradient, Div = -W_p⁻¹ Gᵀ W_v, under
    the r² sinθ volume norm, so grad/div is discretely skew and the
    inviscid spectrum is undamped;
  * Coriolis cross-component interpolations built as adjoint pairs (a
    base interpolation and its volume-weighted transpose); the θ–φ pair is
    exactly skew only with ``coriolis_pairing='adjoint'``, see
    ``staggered_general._Assembly.cross_vector``;
  * pole θ-edges (θ = 0, π) carry zero sinθ weight, so the 2-point θ
    stencil needs no explicit parity fold; higher-order θ stencils use a
    parity-extended node set (``_stag_theta_diff_parity``).

Which systems are built is a property of the dispatcher, not of this
module: any system whose equations classify (evolution, saddle-point
constraint, involution, tensor definition) and whose task kinds all have a
C-grid block, see docs/staggered_general_design.md. Anything else raises
NotImplementedError naming the equation and the supported shapes.

Sign convention: every block is entered as A = −A_phys with ∂t x = A_phys x,
so ω = −iλ; without a Dt term the vector equation is taken as the operator
itself (the Zhang form 2·Cross(Ω, v) − Grad(p) = λ v).

Geometry: shells (r_inner > 0) and balls (r_inner = 0). On a ball the origin
is the inner r-edge, where v_r lives; v_θ, v_φ and the scalars sit on
r-centers strictly inside it. The origin r-edge has zero area and zero
volume weight, so v_r(0, θ) enters no norm-adjoint operator (divergence,
Coriolis partner, entropy source): it is a receive-only ring of Nθ copies
of one physical point. A Cartesian-regular vector has v_r(0, θ) = c·r̂(θ),
whose m-content is: |m| ≥ 2 → 0; m = 0 → c_z cos θ (one DOF); |m| = 1 →
in-plane c ∝ sin θ (one DOF). An inviscid pencil installs these as rows
(``_apply_ball_origin_vr``), a viscous one folds or drops the ring
symmetrically (``_reduce_velocity_dofs``); the radial stencils take the
origin-parity closure f(−r) = ±f(r) (``_origin_parity``). The audit of the
origin treatment and the analytic ball benchmarks are in
docs/ball_staggered.md.

Flat-state layout: every sub-grid is C-ordered with r outer and θ inner
(flat index ir·Nθ + it). Fields have different sizes, so the result carries
a ``StaggeredLayout`` rather than the colocated ``StateLayout``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Staggered sub-grids
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaggeredGrids:
    """The four staggered sub-grids of an Arakawa-C layout on (r, θ).

    Built from a colocated ``Grid`` whose ``r`` are taken as the r-edges
    (inclusive endpoints) and whose ``theta`` are taken as the θ-centers;
    θ-edges are the midpoints between them plus the two poles.

    Attributes
    ----------
    r_edge, r_center : np.ndarray
        Radial edge nodes (N_r) and centers (N_r - 1).
    theta_edge, theta_center : np.ndarray
        Latitudinal edge nodes (N_θ + 1, poles at 0 and π) and centers (N_θ).
    """

    r_edge: np.ndarray
    r_center: np.ndarray
    theta_edge: np.ndarray
    theta_center: np.ndarray

    @property
    def n_center(self) -> int:
        return (len(self.r_center)) * (len(self.theta_center))

    @property
    def n_vr(self) -> int:
        return (len(self.r_edge)) * (len(self.theta_center))

    @property
    def n_vt(self) -> int:
        return (len(self.r_center)) * (len(self.theta_edge))

    @classmethod
    def from_grid(cls, grid) -> "StaggeredGrids":
        r_edge = np.asarray(grid.r, dtype=float)
        r_center = 0.5 * (r_edge[:-1] + r_edge[1:])
        # grid.theta are the θ-centers; the θ-edges are their midpoints with
        # the poles appended so the sinθ volume weight there is exactly zero.
        theta_in = np.asarray(grid.theta, dtype=float)
        theta_edge = np.concatenate((
            [0.0],
            0.5 * (theta_in[:-1] + theta_in[1:]),
            [np.pi],
        )) if len(theta_in) > 1 else np.array([0.0, np.pi])
        theta_center = theta_in.copy()
        return cls(
            r_edge=r_edge,
            r_center=r_center,
            theta_edge=theta_edge,
            theta_center=theta_center,
        )


# ---------------------------------------------------------------------------
# Staggered state layout (per-field sub-grid sizes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaggeredField:
    name: str
    location: str          # 'center', 'r_edge', or 'theta_edge'
    offset: int
    size: int

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.offset + self.size)


@dataclass(frozen=True)
class StaggeredLayout:
    """Maps unknown-name → flat-state slice on the staggered grid.

    Unlike the colocated ``StateLayout``, each field may have a different
    size (centers vs r-edges vs θ-edges). ``placements`` maps each unknown
    to its C-grid grade, 'center', 'face', 'tensor' or 'tensor9'
    (docs/staggered_general_design.md section 2a).
    """

    fields: Tuple[StaggeredField, ...]
    total_size: int
    placements: Optional[Dict[str, str]] = None

    def field(self, name: str) -> StaggeredField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"unknown field {name!r}; have "
                       f"{[f.name for f in self.fields]}")

    def slice_of(self, name: str) -> slice:
        return self.field(name).slice


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StaggeredCompiledMatrices:
    """Output of the staggered compile path.

    Mirrors the colocated ``CompiledMatrices`` interface where meaningful
    (A_total, B_total, m) but carries a ``StaggeredLayout`` and
    ``StaggeredGrids`` instead of the colocated layout/grid. ``constraints``
    is the classification record of docs/staggered_general_design.md
    section 4 and ``b_total`` the source vector (None without source
    terms); for an involution (constrained transport) ``divergence``
    returns the discrete divergence of the flux unknown of a state vector.
    """

    A_total: sp.spmatrix
    B_total: sp.spmatrix
    layout: StaggeredLayout
    grids: StaggeredGrids
    m: int
    discretization: str = "staggered"
    constraints: Optional[dict] = None
    b_total: Optional[np.ndarray] = None

    def divergence(self, x, name=None):
        """The discrete divergence (net flux out of each cell, on the
        centres) of the involution unknown ``name`` (the only one when
        None) in the reduced state vector ``x``."""
        ops = (self.constraints or {}).get("divergence") or {}
        if not ops:
            raise ValueError("this pencil carries no involution unknown")
        if name is None:
            (name,) = ops
        return ops[name] @ np.asarray(x)

    def assemble_evp_matrices(self, mass_equations=None):
        """Return (K, M) for the generalized EVP K x = λ M x.

        Boundary conditions (wall impenetrability, origin regularity) are
        applied during assembly, so ``A_total``/``B_total`` are already the
        final pencil. ``mass_equations`` is accepted for signature
        compatibility with the colocated ``CompiledProblem`` and ignored.
        """
        return self.A_total.tocsr(), self.B_total.tocsr()


# ---------------------------------------------------------------------------
# 1-D staggered stencils
# ---------------------------------------------------------------------------

def _stag_diff_1d(n_face, n_cell, h):
    """2-point staggered first difference cell→face, shape (n_face, n_cell).

    Interior faces difference the two flanking cells; the two end faces use
    a one-sided difference of the two nearest cells.
    """
    D = np.zeros((n_face, n_cell), dtype=complex)
    for i in range(1, n_face - 1):
        D[i, i - 1] = -1.0 / h
        D[i, i] = 1.0 / h
    D[0, 0] = -1.0 / h
    D[0, 1] = 1.0 / h
    D[n_face - 1, n_cell - 2] = -1.0 / h
    D[n_face - 1, n_cell - 1] = 1.0 / h
    return sp.csr_matrix(D)


def _lagrange_on_extension(ext, dst_nodes, order, deriv=1):
    """Centered width-``order`` Lagrange stencil of the ``deriv``-th
    derivative (0 interpolates) on the extended node set ``ext``, a list
    of (coordinate, source column, sign): every extended node feeds its
    real column with its sign. Shape (len(dst), n_columns); the window is
    clamped at the ends of the extension."""
    from spdee.backends.fd2.stencils import _lagrange_derivative_weights
    dst = np.asarray(dst_nodes, dtype=float)
    ext = sorted(ext, key=lambda e: e[0])
    coords = np.array([e[0] for e in ext])
    n = 1 + max(e[1] for e in ext)
    w = min(order, len(ext))
    D = np.zeros((len(dst), n), dtype=complex)
    for i, xe in enumerate(dst):
        j = np.searchsorted(coords, xe)
        j0 = int(np.clip(j - w // 2, 0, len(ext) - w))
        idx = np.arange(j0, j0 + w)
        wts = _lagrange_derivative_weights(coords[idx], float(xe), order=deriv)
        for jj, wt in zip(idx, wts):
            _, col, sign = ext[jj]
            D[i, col] += sign * wt
    return sp.csr_matrix(D)


def _real_nodes(src_nodes):
    src = np.asarray(src_nodes, dtype=float)
    return src, [(src[k], k, 1.0) for k in range(len(src))]


def _stag_theta_diff_parity(src_nodes, dst_nodes, order, parity):
    """Even-order staggered θ first derivative src→dst with pole parity
    extension, shape (len(dst), len(src)).

    Every real source node is reflected across θ=0 (to −θ) and across θ=π
    (to 2π−θ), each reflection mapping back to its real column with sign
    ``parity`` ∈ {±1} (scalars and v_r: (-1)^|m|; v_θ, v_φ: (-1)^|m+1|). A
    centered width-``order`` Lagrange stencil on the extended node set then
    keeps the formal order uniform up to and including the pole-adjacent
    destination points. A node sitting exactly on a pole is its own
    reflection and is not duplicated. At order=2 this reduces to the plain
    2-point staggered difference.
    """
    src, ext = _real_nodes(src_nodes)
    eps = 1e-10
    for k in range(len(src)):
        if src[k] > eps:
            ext.append((-src[k], k, float(parity)))
        if src[k] < np.pi - eps:
            ext.append((2 * np.pi - src[k], k, float(parity)))
    return _lagrange_on_extension(ext, dst_nodes, order)


def _stag_diff_1d_lagrange(src_nodes, dst_nodes, order):
    """Staggered first derivative of even ``order``, src→dst, on a
    centered width-``order`` Lagrange stencil clamped at the ends. The
    composed Div·Grad built from these stencils keeps the stencil order."""
    _, ext = _real_nodes(src_nodes)
    return _lagrange_on_extension(ext, dst_nodes, order)


def _stag_diff_1d_origin_parity(src_nodes, dst_nodes, order, parity):
    """Staggered radial first derivative src→dst with origin parity
    extension, shape (len(dst), len(src)).

    The radial analogue of ``_stag_theta_diff_parity``: the singularity is
    r=0, across which a regular field obeys f(−r) = parity · f(r) with

        parity = (-1)^|m|            for scalars (ρ₁, s₁)
        parity = (-1)^max(|m|-1,0)   for vector components (v_r, v_θ, v_φ, b)

    (see ``_origin_parity``). Every real source node r_k > 0 is reflected to
    −r_k with a parity-signed map back to its column, and a centered
    width-``order`` Lagrange stencil is taken on the extended set, so the
    origin-adjacent rows keep formal order (a one-sided difference at the
    origin face is only O(h) for even-parity fields). On a shell (all r > 0)
    the extension never enters the stencil window. A node exactly at r=0 is
    its own reflection and is not duplicated.
    """
    src, ext = _real_nodes(src_nodes)
    for k in range(len(src)):
        if src[k] > 1e-12:
            ext.append((-src[k], k, float(parity)))
    return _lagrange_on_extension(ext, dst_nodes, order)


def _stag_diff_1d_wall_parity(src_nodes, dst_nodes, order, reflections):
    """Staggered radial first derivative src→dst of even ``order`` with the
    source nodes reflected across the walls in ``reflections``, a list of
    (r_wall, parity): every source node maps to its mirror image 2 r_wall
    - r with sign ``parity`` (−1 for a field vanishing on the wall, the odd
    extension), and a centered width-``order`` Lagrange stencil on the
    extended set keeps its shape up to the wall face, where it is the
    mirror-ghost closure of that order (at order 2 the two-point closure
    ∓2 f_0 / h). The transposed divergence stays consistent through the
    wall cells because every near-wall row is a reflected centered stencil,
    which a one-sided wall row alone is not. Walls without a reflection
    clamp the window as ``_stag_diff_1d_lagrange``.
    """
    src, ext = _real_nodes(src_nodes)
    for r_wall, parity in reflections:
        for k in range(len(src)):
            ext.append((2.0 * r_wall - src[k], k, float(parity)))
    return _lagrange_on_extension(ext, dst_nodes, order)


def _origin_parity(m, kind):
    """Radial origin parity f(−r) = parity · f(r) of a regular field of
    azimuthal order ``m``. ``kind`` ∈ {'scalar', 'vector'}:
        scalar : (-1)^|m|
        vector : (-1)^max(|m|-1,0)   (shared by all three physical components)
    """
    am = abs(int(m))
    if kind == "scalar":
        return (-1) ** (am % 2)
    if kind == "vector":
        return (-1) ** (max(am - 1, 0) % 2)
    raise ValueError(f"kind must be 'scalar' or 'vector'; got {kind!r}")


def _radial_flux_stencils(grids, order, wall_bc, origin_parity):
    """1-D radial stencils of the flux-form scalar Laplacian: the gradient
    centers→edges (Nr_e × Nr_c) and the divergence edges→centers
    (Nr_c × Nr_e).

    Wall faces: ``wall_bc='neumann'`` zeroes the gradient row (no flux);
    ``'dirichlet'`` closes it with the mirror ghost center f_g = −f_0, so the
    face derivative is ∓2 f_0/h over the half cell, and the returned face
    weights carry ½ there so the transposed divergence counts that flux
    once. At order > 2 a Dirichlet wall reflects the source nodes with the
    odd parity (``_stag_diff_1d_wall_parity``) so that every row within the
    stencil window of the wall, the wall row included, is a centered stencil
    on the odd-extended field and the transposed divergence stays consistent
    there; the mirror ghost is second order about the wall whatever the
    stencil order (the even part of f about the wall is misread), so the
    wall cells converge at order 2 while the interior keeps the stencil
    order (docs/staggered_general_design.md section 5). A Neumann wall at
    order > 2 keeps the clamped interior rows with a zero wall row. On a
    ball the inner face is the origin, whose gradient row is the
    origin-parity stencil of ``origin_parity``; that face has zero area, so
    the row never enters the composed operator, but it matches the gradient
    blocks. Interior rows are the 2-point difference at order ≤ 2 and the
    centered Lagrange stencil above; the divergence is −Grᵀ, the negative
    transpose of the gradient, so the flux-form composition is self-adjoint
    under the r² volume weight at every order (at order ≤ 2 this is the
    plain 2-point difference). Returns (Gr, Dr, face_weight) with
    ``face_weight`` (Nr_e,) the wall-face flux weights (1, or ½ on a
    Dirichlet face). ``wall_bc`` is one closure for both walls or an
    (inner, outer) pair.
    """
    r_edge, r_center = grids.r_edge, grids.r_center
    Nr_e, Nr_c = len(r_edge), len(r_center)
    hr = r_edge[1] - r_edge[0]
    is_ball = bool(abs(float(r_edge[0])) < 1e-12)
    inner_bc, outer_bc = (wall_bc, wall_bc) if isinstance(wall_bc, str) else wall_bc
    for bc in (inner_bc, outer_bc):
        if bc not in ("neumann", "dirichlet"):
            raise ValueError(
                f"wall_bc must be 'neumann' or 'dirichlet'; got {bc!r}")
    inner_dirichlet = inner_bc == "dirichlet" and not is_ball
    outer_dirichlet = outer_bc == "dirichlet"
    if order <= 2:
        Gr = np.zeros((Nr_e, Nr_c), dtype=complex)
        for i in range(1, Nr_e - 1):
            Gr[i, i - 1] = -1.0 / hr
            Gr[i, i] = 1.0 / hr
        if outer_dirichlet:
            Gr[Nr_e - 1, Nr_c - 1] = -2.0 / hr
        if inner_dirichlet:
            Gr[0, 0] = 2.0 / hr
    else:
        reflections = ([(r_edge[0], -1)] if inner_dirichlet else []) + (
            [(r_edge[-1], -1)] if outer_dirichlet else [])
        if reflections:
            Gr = _stag_diff_1d_wall_parity(r_center, r_edge, order, reflections).toarray()
        else:
            Gr = _stag_diff_1d_lagrange(r_center, r_edge, order).toarray()
        if not inner_dirichlet:
            Gr[0, :] = 0.0
        if not outer_dirichlet:
            Gr[Nr_e - 1, :] = 0.0
    face_weight = np.ones(Nr_e)
    if outer_dirichlet:
        face_weight[Nr_e - 1] = 0.5
    if inner_dirichlet:
        face_weight[0] = 0.5
    if is_ball:
        Gr[0, :] = _stag_diff_1d_origin_parity(
            r_center, r_edge, max(order, 2), origin_parity).toarray()[0, :]
    Gr = sp.csr_matrix(Gr)
    return Gr, (-Gr.T).tocsr(), face_weight


def _stag_interp_1d_lagrange(src_nodes, dst_nodes, order):
    """Staggered interpolation src→dst of even ``order`` via order-0
    Lagrange weights on a centered width-``order`` window (clamped at the
    ends). ``_avg_1d`` is the order=2 case.

    Used for the base Coriolis cross-component interpolations; their
    partners are the volume-weighted transposes, which inherit the order and
    keep the coupling skew-adjoint.
    """
    _, ext = _real_nodes(src_nodes)
    return _lagrange_on_extension(ext, dst_nodes, order, deriv=0)


def _avg_1d(n_to, n_from):
    """2-point averaging interpolation, shape (n_to, n_from); the end rows
    copy the nearest source value."""
    P = np.zeros((n_to, n_from), dtype=complex)
    for i in range(1, n_to - 1):
        P[i, i - 1] = 0.5
        P[i, i] = 0.5
    P[0, 0] = 1.0
    P[n_to - 1, n_from - 1] = 1.0
    return sp.csr_matrix(P)


def _avg_c2e_zero_ends(n_edge):
    """2-point centers→edges average, shape (n_edge, n_edge - 1), with
    zero end rows (the weightless pole θ-edges)."""
    P = np.zeros((n_edge, n_edge - 1), dtype=complex)
    for j in range(1, n_edge - 1):
        P[j, j - 1] = 0.5
        P[j, j] = 0.5
    return sp.csr_matrix(P)


def _avg_e2c(n_center):
    """2-point edges→centers average, shape (n_center, n_center + 1)."""
    P = np.zeros((n_center, n_center + 1), dtype=complex)
    for j in range(n_center):
        P[j, j] = 0.5
        P[j, j + 1] = 0.5
    return sp.csr_matrix(P)


def _stag_diff_c2e_zero_ends(n_edge, h):
    """2-point centers→edges first difference, shape (n_edge, n_edge - 1),
    with zero end rows (the pole θ-edges, or a radial wall row that a
    closure fills in)."""
    D = np.zeros((n_edge, n_edge - 1), dtype=complex)
    for j in range(1, n_edge - 1):
        D[j, j - 1] = -1.0 / h
        D[j, j] = 1.0 / h
    return sp.csr_matrix(D)


def _stag_diff_e2c(n_center, h):
    """2-point edges→centers first difference, shape (n_center, n_center + 1)."""
    D = np.zeros((n_center, n_center + 1), dtype=complex)
    for j in range(n_center):
        D[j, j] = -1.0 / h
        D[j, j + 1] = 1.0 / h
    return sp.csr_matrix(D)


def _scalar_theta_gradient_1d(grids, m, order):
    """The 1-D θ gradient centers→edges of a scalar: the 2-point difference
    with zero pole rows at order ≤ 2, the parity-extended stencil with the
    scalar parity (-1)^|m| above."""
    th_edge, th_center = grids.theta_edge, grids.theta_center
    if order <= 2:
        ht = th_center[1] - th_center[0] if len(th_center) > 1 else np.pi
        return _stag_diff_c2e_zero_ends(len(th_edge), ht)
    return _stag_theta_diff_parity(th_center, th_edge, order, (-1) ** (abs(m) % 2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compile_fd2_staggered(
    system,
    grid,
    m: int,
    order_r: int = 2,
    order_theta: int = 2,
    coriolis_pairing: str = "center",
    params=None,
    pole_closure: bool = False,
) -> StaggeredCompiledMatrices:
    """Compile an EquationSystem on the staggered (Arakawa-C) grid.

    The task-driven assembler of ``staggered_general`` walks the system,
    classifies every equation (evolution, saddle-point constraint,
    involution, tensor definition) and dispatches every task to a mimetic
    block builder of this module or of ``staggered_mhd``; a system it
    cannot classify is refused with the equation and the supported shapes.

    ``order_r`` / ``order_theta`` (even, default 2) set the staggered stencil
    order. The operators use a single order for both directions, so the
    lower of the two is used when they differ. ``coriolis_pairing``
    ('center', 'solar_eigenproblem' or 'adjoint') selects the v_φ ← v_θ Coriolis block,
    see ``staggered_general._Assembly.cross_vector``. ``params`` binds
    symbolic parameters. ``pole_closure`` selects the pole closure of the
    strain Gram (``build_staggered_weighted_viscous_gram``), off by
    default: it moves the certified m = 8 pencil.
    """
    from spdee.backends.fd2.staggered_general import compile_staggered_general
    return compile_staggered_general(
        system, grid, m, order_r=order_r, order_theta=order_theta,
        params=params, coriolis_pairing=coriolis_pairing, pole_closure=pole_closure)


# ---------------------------------------------------------------------------
# Scalar Laplacians (centers → centers)
# ---------------------------------------------------------------------------

def build_staggered_scalar_laplacian(grids: "StaggeredGrids", m, order=2,
                                     wall_bc="neumann"):
    """Staggered scalar Laplacian ∇² mapping cell centers → cell centers.

    Built in flux form as the composition of the staggered gradient
    (centers→edges) with the flux-form divergence (edges→centers):

        ∇²f = (1/r²) ∂_r(r² ∂_r f)
            + (1/(r² sinθ)) ∂_θ(sinθ ∂_θ f)
            - m²/(r² sin²θ) · f

    The composed operator is discretely self-adjoint under the volume norm
    (Div is the norm-adjoint of Grad), so −∇² has a real positive spectrum;
    a direct second-derivative stencil of the advective-form metric terms
    (2/r ∂_r, cotθ/r² ∂_θ) is not self-adjoint and injects spurious complex
    frequencies into an eigenproblem. The radial wall rows of the gradient
    come from ``_radial_flux_stencils``: zero for ``wall_bc='neumann'``
    (natural, ∂_r f = 0), the mirror-ghost closure for ``'dirichlet'``
    (f = 0 on the wall); on a ball the inner face is the origin (zero area,
    origin-parity row).

    ``order`` (even, default 2) is the order of the first-derivative
    stencils, which the composition preserves. At order > 2 the radial
    stencils are ``_stag_diff_1d_lagrange`` and the θ stencils are the
    parity-extended ``_stag_theta_diff_parity`` with the scalar parity
    (-1)^|m|.

    Returns an (n_center × n_center) sparse operator.
    """
    r_edge, r_center = grids.r_edge, grids.r_center
    th_edge, th_center = grids.theta_edge, grids.theta_center
    Nr_c, Nt_c = len(r_center), len(th_center)

    def diagc(rv, tv, fn):
        return sp.diags(fn(rv[:, None], tv[None, :]).ravel().astype(complex))

    # radial gradient centers→edges and divergence edges→centers
    Gr_1d, Dr_1d, face_w = _radial_flux_stencils(grids, order, wall_bc,
                                                 _origin_parity(m, "scalar"))
    Gr = sp.kron(Gr_1d, sp.eye(Nt_c), format="csr")
    Dr = sp.kron(Dr_1d, sp.eye(Nt_c), format="csr")

    # θ gradient centers→edges and its negative transpose
    Gt_1d = _scalar_theta_gradient_1d(grids, m, order)
    Dt_1d = (-Gt_1d.T).tocsr()
    Gt = sp.kron(sp.eye(Nr_c), Gt_1d, format="csr")
    Dt = sp.kron(sp.eye(Nr_c), Dt_1d, format="csr")

    r2e = diagc(r_edge, th_center, lambda r, t: r**2 * face_w[:, None] + 0 * t)
    inv_r2c = diagc(r_center, th_center, lambda r, t: 1.0 / r**2 + 0 * t)
    sin_te = diagc(r_center, th_edge, lambda r, t: np.sin(t) + 0 * r)
    inv_r2sin = diagc(r_center, th_center, lambda r, t: 1.0 / (r**2 * np.sin(t)))
    mterm = diagc(r_center, th_center,
                  lambda r, t: -(m**2) / (r**2 * np.sin(t)**2) + 0j)

    return (inv_r2c @ Dr @ r2e @ Gr
            + inv_r2sin @ Dt @ sin_te @ Gt
            + mterm).tocsr()


def _robin_wall_term(grids: "StaggeredGrids", ghost: dict, weight_fn=None):
    """The wall flux of a Robin closure on the flux-form scalar Laplacian.

    ``ghost`` maps a wall (0 inner, 1 outer) to the ghost coefficient g
    per theta centre: the wall face flux is g f_0 with f_0 the wall cell
    value (``staggered_general._composite_closure``, the mirror ghost of
    c_N d_r f + c_D f = 0). The flux-form divergence counts that flux once,
    so the wall cell's diagonal gains sigma r_w^2 g / (r_0^2 h), sigma = +1
    at the outer wall and -1 at the inner, times W(r_w) / W(r_0) under a
    weight (the same insertion as ``build_staggered_weighted_scalar_laplacian``);
    the closure's flux is built on the natural (zero-flux) stencils of that
    wall. Returns the diagonal on the centres."""
    r_edge, r_center, th_center = grids.r_edge, grids.r_center, grids.theta_center
    Nr_c, Nt_c = len(r_center), len(th_center)
    hr = r_edge[1] - r_edge[0]
    diag = np.zeros(Nr_c * Nt_c, dtype=complex)
    for wall, g in ghost.items():
        i, j0, sigma = (len(r_edge) - 1, Nr_c - 1, 1.0) if wall == 1 else (0, 0, -1.0)
        rw, r0 = r_edge[i], r_center[j0]
        vals = sigma * rw**2 * np.asarray(g, dtype=complex) / (r0**2 * hr)
        if weight_fn is not None:
            ww = np.asarray(weight_fn(np.full_like(th_center, rw), th_center))
            w0 = np.asarray(weight_fn(np.full_like(th_center, r0), th_center))
            vals = vals * ww / w0
        diag[j0 * Nt_c:(j0 + 1) * Nt_c] = vals
    return sp.diags(diag, format="csr")


def build_staggered_angular_laplacian(grids: "StaggeredGrids", m, order=2):
    """The angular part of ``build_staggered_scalar_laplacian``,
    (1/(r² sinθ)) ∂_θ(sinθ ∂_θ f) − m²/(r² sin²θ) f, centers → centers,
    with the same θ stencils (self-adjoint under the volume norm)."""
    r_center = grids.r_center
    th_edge, th_center = grids.theta_edge, grids.theta_center
    Nr_c = len(r_center)

    def diagc(rv, tv, fn):
        return sp.diags(fn(rv[:, None], tv[None, :]).ravel().astype(complex))

    Gt_1d = _scalar_theta_gradient_1d(grids, m, order)
    Dt_1d = (-Gt_1d.T).tocsr()
    Gt = sp.kron(sp.eye(Nr_c), Gt_1d, format="csr")
    Dt = sp.kron(sp.eye(Nr_c), Dt_1d, format="csr")
    sin_te = diagc(r_center, th_edge, lambda r, t: np.sin(t) + 0 * r)
    inv_r2sin = diagc(r_center, th_center, lambda r, t: 1.0 / (r**2 * np.sin(t)))
    mterm = diagc(r_center, th_center,
                  lambda r, t: -(m**2) / (r**2 * np.sin(t)**2) + 0j)
    return (inv_r2sin @ Dt @ sin_te @ Gt + mterm).tocsr()


def build_staggered_weighted_scalar_laplacian(grids, m, weight_fn, order=2,
                                              wall_bc="neumann"):
    """Weighted mimetic scalar Laplacian

        L_w f = (1/W) ∇·(W ∇f) = ∇²f + ∇ln W · ∇f

    for a positive background weight W(r, θ) (e.g. P₀ for the solar eigenproblem's thermal
    diffusion (1/(ρ₀T₀))∇·(κρ₀T₀∇s₁) = κ(1/P₀)∇·(P₀∇s₁), ideal gas
    ρ₀T₀ ∝ P₀). Built by inserting diag(W) on the edge flux grids and
    diag(1/W) on centers around the same flux-form Div·Grad stencils as
    ``build_staggered_scalar_laplacian``.

    This divergence-form operator is self-adjoint and negative-definite
    under the W-weighted center volume norm, since the W-weighted Div is the
    norm-adjoint of Grad; the explicit split ∇²f + (∇ln W)·∇f with a
    one-sided advective correction is not. It reduces to the plain scalar
    Laplacian for constant W. The azimuthal −m²/(r² sin²θ) term is algebraic
    (carries no flux) and enters unweighted.

    ``weight_fn(r, θ)`` must accept broadcast 2-D arrays. ``order`` and
    ``wall_bc`` ('neumann' / 'dirichlet') thread through the same radial
    stencils as ``build_staggered_scalar_laplacian`` (origin-parity row at
    the ball origin, whose face has zero area).
    """
    r_edge, r_center = grids.r_edge, grids.r_center
    th_edge, th_center = grids.theta_edge, grids.theta_center
    Nr_c, Nt_c = len(r_center), len(th_center)

    def diagc(rv, tv, fn):
        return sp.diags(np.asarray(fn(rv[:, None], tv[None, :])).ravel()
                        .astype(complex))

    Gr_1d, Dr_1d, face_w = _radial_flux_stencils(grids, order, wall_bc,
                                                 _origin_parity(m, "scalar"))
    Gr = sp.kron(Gr_1d, sp.eye(Nt_c), format="csr")
    Dr = sp.kron(Dr_1d, sp.eye(Nt_c), format="csr")

    Gt_1d = _scalar_theta_gradient_1d(grids, m, order)
    Dt_1d = (-Gt_1d.T).tocsr()
    Gt = sp.kron(sp.eye(Nr_c), Gt_1d, format="csr")
    Dt = sp.kron(sp.eye(Nr_c), Dt_1d, format="csr")

    r2e = diagc(r_edge, th_center, lambda r, t: r**2 * face_w[:, None] + 0 * t)
    inv_r2c = diagc(r_center, th_center, lambda r, t: 1.0 / r**2 + 0 * t)
    sin_te = diagc(r_center, th_edge, lambda r, t: np.sin(t) + 0 * r)
    inv_r2sin = diagc(r_center, th_center, lambda r, t: 1.0 / (r**2 * np.sin(t)))
    mterm = diagc(r_center, th_center,
                  lambda r, t: -(m**2) / (r**2 * np.sin(t)**2) + 0j)

    # weight on the two edge flux grids and its inverse on centers
    W_re = diagc(r_edge, th_center, weight_fn)
    W_te = diagc(r_center, th_edge, weight_fn)
    inv_W_c = diagc(r_center, th_center,
                    lambda r, t: 1.0 / np.asarray(weight_fn(r, t)))

    return (inv_W_c @ inv_r2c @ Dr @ r2e @ W_re @ Gr
            + inv_W_c @ inv_r2sin @ Dt @ sin_te @ W_te @ Gt
            + mterm).tocsr()


# ---------------------------------------------------------------------------
# Gradient / divergence pair and wall DOF elimination
# ---------------------------------------------------------------------------

def _build_staggered_gradient_blocks(grids, m, order):
    """The three staggered gradient blocks (G_r, G_t, G_p) of a center scalar:

    G_r = ∂_r p          centers → r-edges  (origin-parity stencil on a ball)
    G_t = (1/r)∂_θ p     centers → θ-edges  (scalar pole parity (-1)^|m|)
    G_p = (im/(r sinθ))p centers → centers  (algebraic)

    Shared by every consumer so the gradient is built one way and the
    divergence is always its norm-adjoint transpose.
    """
    r_edge, r_center = grids.r_edge, grids.r_center
    th_edge, th_center = grids.theta_edge, grids.theta_center
    Nr_e, Nr_c = len(r_edge), len(r_center)
    Nt_c = len(th_center)
    hr = r_edge[1] - r_edge[0]
    im = 1j * m
    I_tc = sp.eye(Nt_c, format="csr", dtype=complex)
    I_rc = sp.eye(Nr_c, format="csr", dtype=complex)

    def diagrc(rv, tv, fn):
        return sp.diags(np.asarray(fn(rv[:, None], tv[None, :])).ravel()
                        .astype(complex), format="csr")

    # radial gradient centers→r-edges
    if bool(abs(float(r_edge[0])) < 1e-12):
        Gr_1d = _stag_diff_1d_origin_parity(
            r_center, r_edge, max(order, 2), _origin_parity(m, "scalar"))
    elif order <= 2:
        Gr_1d = _stag_diff_1d(Nr_e, Nr_c, hr)
    else:
        Gr_1d = _stag_diff_1d_lagrange(r_center, r_edge, order)
    G_r = sp.kron(Gr_1d, I_tc, format="csr")

    # θ gradient centers→θ-edges (pole rows zero at order ≤ 2)
    Gt_1d = _scalar_theta_gradient_1d(grids, m, order)
    inv_r_vt = diagrc(r_center, th_edge, lambda r, t: 1.0 / r + 0 * t)
    G_t = inv_r_vt @ sp.kron(I_rc, Gt_1d, format="csr")

    # φ algebraic term (no derivative; lives on centers with p)
    G_p = diagrc(r_center, th_center, lambda r, t: im / (r * np.sin(t)))
    return G_r, G_t, G_p


def build_staggered_grad_div_pair(grids: "StaggeredGrids", m, order=2):
    """Return (G, Div, W_v, W_p): the staggered gradient (centers → stacked
    velocity state [v_r; v_θ; v_φ]), the norm-adjoint divergence
    (velocity → centers), and the diagonal volume-weight norms on the
    velocity state and the center grid.

    Div = -W_p⁻¹ Gᵀ W_v is the exact norm-adjoint of G and equals the
    flux-form divergence in the interior. Because G's wall-edge rows are
    one-sided stencils, the transposed rows are not a consistent divergence
    in the ~3 centers adjacent to each radial wall (the discrete divergence
    theorem and pointwise wall consistency cannot both hold without a
    surface term ∮ p v·n). With the wall v_r DOFs removed as rows AND
    columns (``_eliminate_wall_vr``; impenetrability v_r = 0 makes the
    surface term vanish) the composed grad-div is consistent to 2nd order
    including the wall-adjacent rows and self-adjoint negative-semidefinite
    on the constrained space; an inviscid pencil overwrites the wall v_r
    rows instead (``staggered_general._apply_gauge_and_walls``), which
    removes the spurious term there.

    ``order`` (even, default 2) sets the stencil order; the divergence
    inherits it (and the θ pole parity) through the transpose.
    """
    r_edge, r_center = grids.r_edge, grids.r_center
    th_edge, th_center = grids.theta_edge, grids.theta_center
    hr = r_edge[1] - r_edge[0]
    ht = th_center[1] - th_center[0] if len(th_center) > 1 else np.pi

    G_r, G_t, G_p = _build_staggered_gradient_blocks(grids, m, order)
    G = sp.vstack([G_r, G_t, G_p], format="csr")

    def vol(rv, tv):
        return ((rv**2)[:, None] * np.sin(tv)[None, :]).ravel().astype(complex) \
            * hr * ht
    W_vr = vol(r_edge, th_center)
    W_vt = vol(r_center, th_edge)
    W_vp = vol(r_center, th_center)
    W_p = vol(r_center, th_center)
    W_v = np.concatenate([W_vr, W_vt, W_vp])
    Div = -(sp.diags(1.0 / W_p) @ G.conj().T @ sp.diags(W_v))
    return G.tocsr(), Div.tocsr(), W_v, W_p


def _eliminate_wall_vr(op, grids):
    """Impenetrable (v_r = 0) radial-wall condition on an operator over the
    stacked velocity state [v_r; v_θ; v_φ]: the wall v_r DOFs (r-edges 0
    and Nr_e - 1 of the leading block) are removed as both rows and
    columns, which keeps a norm-self-adjoint operator self-adjoint on the
    constrained space (zeroing the input alone leaves spurious transpose
    column couplings). Returns (reduced_op, keep_mask)."""
    Nr_e, Nt_c = len(grids.r_edge), len(grids.theta_center)
    keep = np.ones(op.shape[0], dtype=bool)
    keep[:Nt_c] = False
    keep[(Nr_e - 1) * Nt_c:Nr_e * Nt_c] = False
    op = op.tocsr()
    return op[keep][:, keep], keep


# ---------------------------------------------------------------------------
# Curl, curl-curl and the vector Laplacian
# ---------------------------------------------------------------------------

def _inv_sin_with_pole_closure(dst_th, order, n_r):
    """Operator S (n_r·n_dst × n_r·n_dst) mapping a field N on the
    r × dst_th grid to (1/sinθ)·N with a L'Hôpital closure at the pole nodes
    of dst_th.

    Away from the poles S is diag(1/sinθ). At a pole node N is a 0/0
    numerator with N(pole) = 0, and (1/sinθ)N → N'(θ)/cosθ; N'(pole) is a
    one-sided width-``order`` Lagrange derivative over the real dst nodes,
    oriented away from the pole (forward at θ≈0, backward at θ≈π), so the
    two pole closures are exact mirror images. A centered parity-extended
    window would fold in a reflected node with a flipped sign and point the
    stencil the wrong way at θ=0. Acts blockwise over the n_r radial nodes.
    """
    from spdee.backends.fd2.stencils import _lagrange_derivative_weights
    dst = np.asarray(dst_th, float)
    n = len(dst)
    sin_d = np.sin(dst)
    cos_d = np.cos(dst)
    S1 = sp.lil_matrix((n, n), dtype=complex)
    for i in range(n):
        if abs(sin_d[i]) > 1e-12:
            S1[i, i] = 1.0 / sin_d[i]
        else:
            w = min(order, n)
            if dst[i] < np.pi / 2:        # near θ=0: forward window
                idx = np.arange(0, w)
            else:                          # near θ=π: backward window
                idx = np.arange(n - w, n)
            wts = _lagrange_derivative_weights(dst[idx], float(dst[i]),
                                               order=1) / cos_d[i]
            for jj, wt in zip(idx, wts):
                if abs(wt) > 0:
                    S1[i, jj] = wt
    return sp.kron(sp.eye(n_r), S1.tocsr(), format="csr")


@dataclass(frozen=True)
class TangentialWall:
    """The tangential wall closure of a face unknown as a Navier slip
    length per wall, v_t + ls (d_r v_t - v_t / r) = 0: ls = 0 is no-slip
    (v_t = 0), ls = inf is stress-free (d_r(v_t / r) = 0) and a finite
    ls a partial slip. ``slip`` holds (inner, outer); each is a number
    or a pair (edges, centers) of tuples with one value per theta edge
    (the closure of e_rtheta on the corners) and per theta centre (that
    of e_rphi on the r-faces). ``open`` flags the walls whose v_r faces
    stay degrees of freedom (a normal-stress wall): the closure then
    keeps the theta derivative of v_r in the wall shear, scaled by the
    same ghost relation. The strings 'noslip' and 'stress_free' are the
    limits with closed walls (``normalize_wall_bc``)."""
    slip: tuple
    open: tuple = (False, False)

    def lengths(self, wall: int, kind: str, n: int) -> np.ndarray:
        """The slip lengths at wall 0 (inner) or 1 (outer) on the n theta
        ``kind`` nodes ('edges' or 'centers')."""
        ls = self.slip[wall]
        if isinstance(ls, tuple):
            ls = ls[0 if kind == "edges" else 1]
        return np.full(n, float(ls)) if np.ndim(ls) == 0 else np.asarray(ls, float)

    def constant(self, wall: int) -> float:
        """One slip length for a wall, when it is constant along it."""
        ls = self.slip[wall]
        if isinstance(ls, tuple):
            vals = np.concatenate([np.asarray(v, float) for v in ls])
            if vals.size == 0:
                return float("inf")
            if not np.all(vals == vals[0]):
                raise NotImplementedError(
                    "staggered: the curl closure of a slip wall takes one slip "
                    "length along the wall; this one varies with theta")
            return float(vals[0])
        return float(ls)


def normalize_wall_bc(wall_bc):
    """The ``TangentialWall`` of a builder's ``wall_bc`` argument: None
    stays None (plain one-sided rows), 'noslip' and 'stress_free' are the
    ls = 0 and ls = inf closures, a ``TangentialWall`` is returned as is."""
    if wall_bc is None or isinstance(wall_bc, TangentialWall):
        return wall_bc
    if wall_bc == "noslip":
        return TangentialWall((0.0, 0.0))
    if wall_bc == "stress_free":
        return TangentialWall((float("inf"), float("inf")))
    raise ValueError(f"unknown wall_bc {wall_bc!r}")


def _slip_ratio(ls, rw, r0, hr):
    """rho = v_g / v_0 of the Navier-slip ghost at a wall face rw with the
    wall cell at r0 and the mirror ghost at rg = 2 rw - r0 (hr = |rg -
    r0|): v_t + ls (d_r v_t - v_t / r) = 0 with the wall value (v_0 +
    v_g) / 2, the derivative (v_g - v_0) / (rg - r0) and v_t / r at the
    wall, the spelling of the ball strain and of the curl closures.
    ls = 0 gives -1 exactly, ls = inf gives rg / r0."""
    ls = np.asarray(ls, float)
    s = np.sign(rw - r0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(np.isinf(ls), (2.0 * rw - r0) / r0,
                       (s * ls / hr + ls / (2.0 * rw) - 0.5)
                       / (s * ls / hr - ls / (2.0 * rw) + 0.5))
    return rho


def _slip_ratio_divided(ls, rw, r0, hr):
    """gamma = g_g / g_0 of the same closure on the divided field
    g = v_t / r (the shell strain's spelling r d_r(v_t / r)): the wall
    value (r0 g_0 + rg g_g) / 2, the shear rw (g_g - g_0) / (rg - r0).
    ls = 0 gives -r0 / rg exactly, ls = inf gives 1."""
    ls = np.asarray(ls, float)
    rg = 2.0 * rw - r0
    s = np.sign(rw - r0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(np.isinf(ls), 1.0,
                         (2.0 * s * ls * rw / hr - r0) / (2.0 * s * ls * rw / hr + rg))
    return gamma


def _slip_wall_shear_factor(ls, rw, r0, hr, divided):
    """The factor the slip ghost leaves on the v_r part of the wall shear
    ((1 / r) d_theta v_r, (i m / (r sin theta)) v_r, nonzero on an open
    wall): the tangential condition reads the whole shear, so the ghost
    absorbs a share of it; 1 at ls = 0, 0 at ls = inf."""
    ls = np.asarray(ls, float)
    s = np.sign(rw - r0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if divided:
            rg = 2.0 * rw - r0
            f = rg / (rg + 2.0 * s * ls * rw / hr)
        else:
            f = 1.0 / (1.0 + 2.0 * s * ls / hr - ls / rw)
    return np.where(np.isinf(ls), 0.0, f)


def _slip_face_weight(ls, hr):
    """The tensor-norm weight of a wall shear face under the slip
    closure, 1/2 + ls / hr: the ghost row's coefficient counts the wall
    stress (1 + r0 / rg) / hr times over the flux form's 1 / hr at ls =
    0, where the half weight is the certified no-slip closure, and once
    as ls grows; 1 where the row is empty (ls = inf)."""
    ls = np.asarray(ls, float)
    return np.where(np.isinf(ls), 1.0, 0.5 + ls / hr)


def _stag_diff_1d_centers_to_edges_bc(src_r, dst_r, wall_bc):
    """2nd-order centers→edges radial first difference with a BC-aware ghost
    closure at the two wall faces dst_r[0] and dst_r[-1].

    Interior faces use the centered difference of the two flanking centers.
    At a wall face the derivative is (f0 − f_ghost)/(r0 − rg), where f0 is
    the first interior center value at radius r0 and the ghost center sits at
    rg = 2·rw − r0 (the mirror of r0 about the wall radius rw). The quantity
    differentiated is the flux f = r·v_tan (v_tan = v_θ or v_φ), and the ghost
    value follows from the velocity condition:
      - 'noslip'      : v_tan(wall) = 0 ⇒ v_tan odd about rw ⇒
                        f_ghost = −(rg/r0)·f0
      - 'stress_free' : ∂_r(v_tan/r) = 0 ⇒ v_tan/r even about rw ⇒
                        f_ghost = (rg/r0)²·f0
      - a ``TangentialWall`` with a finite slip length ls (one value along
        the wall): v_ghost = rho v0 with rho of ``_slip_ratio`` ⇒
                        f_ghost = rho (rg/r0)·f0
    The ghost is folded back onto the real column of f0.
    """
    src = np.asarray(src_r, float)
    dst = np.asarray(dst_r, float)
    n_cell, n_face = len(src), len(dst)
    assert n_cell == n_face - 1, "centers→edges expects n_cell = n_face-1"
    D = sp.lil_matrix((n_face, n_cell), dtype=complex)
    for i in range(1, n_face - 1):
        h = src[i] - src[i - 1]
        D[i, i - 1] = -1.0 / h
        D[i, i] = 1.0 / h
    wall_bc = normalize_wall_bc(wall_bc)
    hr = dst[1] - dst[0]

    def ghost_coeff(rw, r0, ls):
        rg = 2.0 * rw - r0          # ghost center radius
        if abs(rw) < 1e-12:
            return 0.0              # the ball origin: the parity row replaces it
        if ls == 0.0:
            g = -(rg / r0)          # f_ghost = g * f0
        elif np.isinf(ls):
            g = (rg / r0) ** 2
        else:
            g = float(_slip_ratio(ls, rw, r0, hr)) * (rg / r0)
        return (1.0 - g) / (r0 - rg)

    D[0, 0] = ghost_coeff(dst[0], src[0], wall_bc.constant(0))
    D[n_face - 1, n_cell - 1] = ghost_coeff(dst[-1], src[-1], wall_bc.constant(1))
    return sp.csr_matrix(D)


def _curl_radial_diff(src_r, dst_r, order, wall_bc=None, origin_parity=None):
    """1-D radial first-derivative stencil src_r → dst_r.

    ``origin_parity`` ∈ {+1, −1, None}: when either node set includes the
    ball origin r=0 and a parity is given, the origin-parity-folded stencil
    ``_stag_diff_1d_origin_parity`` is used (pass the SOURCE field's origin
    parity) so the origin-adjacent rows keep formal order; its outer-wall
    row still takes the ``wall_bc`` ghost closure below. Otherwise, at
    order ≤ 2: centers→edges uses the BC-aware ghost closure at the wall
    faces when ``wall_bc`` ('noslip' / 'stress_free') is given, else the
    plain one-sided wall stencil; edges→centers is the plain 2-point
    difference. Any other case uses the centered Lagrange stencil.
    """
    includes_origin = (len(dst_r) > 0 and abs(float(dst_r[0])) < 1e-12) or \
                      (len(src_r) > 0 and abs(float(src_r[0])) < 1e-12)
    if origin_parity is not None and includes_origin:
        D = _stag_diff_1d_origin_parity(src_r, dst_r, max(order, 2),
                                        origin_parity)
        if wall_bc is not None and order <= 2 and len(src_r) == len(dst_r) - 1:
            Dbc = _stag_diff_1d_centers_to_edges_bc(src_r, dst_r, wall_bc).tolil()
            D = D.tolil()
            D.rows[-1] = Dbc.rows[-1]
            D.data[-1] = Dbc.data[-1]
            D = D.tocsr()
        return D
    if order <= 2 and len(src_r) == len(dst_r) - 1:
        # centers→edges
        if wall_bc is not None:
            return _stag_diff_1d_centers_to_edges_bc(src_r, dst_r, wall_bc)
        return _stag_diff_1d(len(dst_r), len(src_r), src_r[1] - src_r[0]
                             if len(src_r) > 1 else 1.0)
    if order <= 2 and len(src_r) == len(dst_r) + 1:
        return _stag_diff_e2c(len(dst_r), src_r[1] - src_r[0])
    return _stag_diff_1d_lagrange(src_r, dst_r, order)


def build_staggered_curl(grids, m, order=2, dual=False, wall_bc=None):
    """Staggered curl on the Arakawa-C grid as six shape-checked blocks.

    Sub-grids (r-index × θ-index):
        RC = (r_edge,   θ_center)   v_r          curl_θ
        CE = (r_center, θ_edge)     v_θ          curl_r
        CC = (r_center, θ_center)   v_φ, p
        EE = (r_edge,   θ_edge)     curl_φ

    ``dual=False`` (primal): input v on {v_r:RC, v_θ:CE, v_φ:CC}, output
    curl(v) on {curl_r:CE, curl_θ:RC, curl_φ:EE}.
    ``dual=True``: input a on {a_r:CE, a_θ:RC, a_φ:EE}, output on
    {RC, CE, CC}. Same differential curl; only the grids swap (the C-grid
    is self-dual), so primal then dual gives curl(curl).

    Components (∂_φ → im):
        curl_r = (1/(r sinθ)) [∂_θ(sinθ v_φ) − im v_θ]
        curl_θ = (im/(r sinθ)) v_r − (1/r) ∂_r(r v_φ)
        curl_φ = (1/r) [∂_r(r v_θ) − ∂_θ v_r]

    Pole treatment: θ-derivatives use the parity-extended stencils with the
    SOURCE field's parity (v_r, curl_θ: (-1)^|m|; v_θ, v_φ, curl_r, curl_φ:
    (-1)^|m+1|); the differentiated flux sinθ·v_φ carries the opposite
    parity since sinθ is odd. On the pole θ-edges the 1/sinθ factors are a
    0/0 limit with a finite value, closed by ``_inv_sin_with_pole_closure``
    (numerator built with the flux parity so it vanishes at the pole; then
    N'(θ)/cosθ from a one-sided derivative away from the pole). 1/(r sinθ)
    factors on grids that do not need a closure are zero-guarded at the
    poles and at the ball origin r-edge, which carry zero norm weight.

    ``wall_bc`` (None / 'stress_free' / 'noslip') selects the ghost closure
    of the radial derivatives of r·v_tan at the wall faces (primal only). On
    a ball the radial derivatives of r·v_tan use the origin-parity stencil
    with parity −(vector parity), since the r factor is odd.

    Returns a dict of the six named blocks (cr_aphi, cr_ath, cth_ar,
    cth_aphi, cphi_ath, cphi_ar) plus the input/output grid shapes under
    "_src" / "_dst".
    """
    re, rc = grids.r_edge, grids.r_center
    te, tc = grids.theta_edge, grids.theta_center
    Nr_e, Nr_c, Nt_e, Nt_c = len(re), len(rc), len(te), len(tc)
    im = 1j * m
    par_s = (-1) ** (abs(m) % 2)          # scalar / v_r / curl_θ parity
    par_v = (-1) ** ((abs(m) + 1) % 2)    # v_θ, v_φ, curl_r, curl_φ parity
    _is_ball_curl = bool(abs(float(re[0])) < 1e-12)
    _op_vec = _origin_parity(m, "vector") if _is_ball_curl else None
    _op_rflux_vec = (-_op_vec) if _op_vec is not None else None   # parity of r·v_vec

    def diag(rv, tv, fn):
        return sp.diags(np.asarray(fn(rv[:, None], tv[None, :])).ravel()
                        .astype(complex), format="csr")

    def inv_r_origin_guarded(rv, tv):
        # 1/r with a zero at the (weightless) ball origin r-edge
        def fn(r, t):
            nz = np.abs(r) > 1e-14
            return np.where(nz, 1.0 / np.where(nz, r, 1.0), 0.0) + 0 * t
        return diag(rv, tv, fn)

    def inv_rsin_origin_guarded(rv, tv):
        # 1/(r sinθ) with zeros at the ball origin r-edge and at the poles
        def fn(r, t):
            s = np.sin(t)
            nzs = np.abs(s) > 1e-14
            nzr = np.abs(r) > 1e-14
            denom = np.where(nzr, r, 1.0) * np.where(nzs, s, 1.0)
            return np.where(nzr & nzs, 1.0 / denom, 0.0)
        return diag(rv, tv, fn)

    # grid shapes of the three input and three output components
    if not dual:
        src = {"r": (Nr_e, Nt_c), "th": (Nr_c, Nt_e), "ph": (Nr_c, Nt_c)}
        dst = {"r": (Nr_c, Nt_e), "th": (Nr_e, Nt_c), "ph": (Nr_e, Nt_e)}
    else:
        src = {"r": (Nr_c, Nt_e), "th": (Nr_e, Nt_c), "ph": (Nr_e, Nt_e)}
        dst = {"r": (Nr_e, Nt_c), "th": (Nr_c, Nt_e), "ph": (Nr_c, Nt_c)}

    def blk(M_, dst_key, src_key):
        dn = dst[dst_key][0] * dst[dst_key][1]
        sn = src[src_key][0] * src[src_key][1]
        assert M_.shape == (dn, sn), (
            f"block dst={dst_key}{dst[dst_key]} src={src_key}{src[src_key]} "
            f"expected {(dn, sn)} got {M_.shape}")
        return M_

    blocks = {}
    if not dual:
        # curl_r on CE: N = ∂_θ(sinθ v_φ) − im v_θ built on CE (flux parity
        # −par_v for the differentiated sinθ·v_φ), then the pole-closed
        # (1/sinθ) and 1/r.
        Dt_flux = _stag_theta_diff_parity(tc, te, order, -par_v)
        sin_cc = diag(rc, tc, lambda r, t: np.sin(t) + 0 * r)
        N_from_vphi = sp.kron(sp.eye(Nr_c), Dt_flux) @ sin_cc      # CC->CE
        N_from_vth = -im * sp.eye(Nr_c * Nt_e, format="csr")       # CE->CE
        inv_r_ce = diag(rc, te, lambda r, t: 1.0 / r + 0 * t)
        S_ce = _inv_sin_with_pole_closure(te, order, Nr_c)
        blocks["cr_aphi"] = blk(inv_r_ce @ S_ce @ N_from_vphi, "r", "ph")
        blocks["cr_ath"] = blk(inv_r_ce @ S_ce @ N_from_vth, "r", "th")
        # curl_θ on RC: (im/(r sinθ)) v_r − (1/r) ∂_r(r v_φ)
        inv_rsin_rc = inv_rsin_origin_guarded(re, tc)
        inv_r_rc = inv_r_origin_guarded(re, tc)
        r_cc = diag(rc, tc, lambda r, t: r + 0 * t)
        Dr_cc2rc = _curl_radial_diff(rc, re, order, wall_bc=wall_bc,
                                     origin_parity=_op_rflux_vec)  # ∂_r(r v_φ) CC->RC
        blocks["cth_ar"] = blk(im * inv_rsin_rc, "th", "r")
        blocks["cth_aphi"] = blk(
            -inv_r_rc @ sp.kron(Dr_cc2rc, sp.eye(Nt_c)) @ r_cc, "th", "ph")
        # curl_φ on EE: (1/r) [∂_r(r v_θ) − ∂_θ v_r]
        inv_r_ee = inv_r_origin_guarded(re, te)
        r_ce = diag(rc, te, lambda r, t: r + 0 * t)
        Dr_ce2ee = _curl_radial_diff(rc, re, order, wall_bc=wall_bc,
                                     origin_parity=_op_rflux_vec)  # ∂_r(r v_θ) CE->EE
        Dt_rc2ee = _stag_theta_diff_parity(tc, te, order, par_s)  # ∂_θ v_r RC->EE
        blocks["cphi_ath"] = blk(
            inv_r_ee @ sp.kron(Dr_ce2ee, sp.eye(Nt_e)) @ r_ce, "ph", "th")
        blocks["cphi_ar"] = blk(
            -inv_r_ee @ sp.kron(sp.eye(Nr_e), Dt_rc2ee), "ph", "r")
    else:
        # out_r on RC: (1/(r sinθ)) [∂_θ(sinθ a_φ) − im a_θ]. RC θ-nodes are
        # centers (never at a pole), so 1/sinθ is finite; the differentiated
        # flux sinθ·a_φ still uses the flux parity −par_v.
        Dt_flux = _stag_theta_diff_parity(te, tc, order, -par_v)  # EE->RC
        sin_ee = diag(re, te, lambda r, t: np.sin(t) + 0 * r)
        inv_rsin_rc = inv_rsin_origin_guarded(re, tc)
        blocks["cr_aphi"] = blk(
            inv_rsin_rc @ sp.kron(sp.eye(Nr_e), Dt_flux) @ sin_ee, "r", "ph")
        blocks["cr_ath"] = blk(-im * inv_rsin_rc, "r", "th")
        # out_θ on CE: (im/(r sinθ)) a_r − (1/r) ∂_r(r a_φ). CE includes the
        # pole θ-edges, where im a_r/sinθ is 0/0: pole closure with the
        # parity of a_r = curl_r (par_v).
        inv_r_ce = diag(rc, te, lambda r, t: 1.0 / r + 0 * t)
        r_ee = diag(re, te, lambda r, t: r + 0 * t)
        Dr_ee2ce = _curl_radial_diff(re, rc, order)  # ∂_r EE->CE
        S_ce_ar = _inv_sin_with_pole_closure(te, order, Nr_c)
        blocks["cth_ar"] = blk(im * inv_r_ce @ S_ce_ar, "th", "r")
        blocks["cth_aphi"] = blk(
            -inv_r_ce @ sp.kron(Dr_ee2ce, sp.eye(Nt_e)) @ r_ee, "th", "ph")
        # out_φ on CC: (1/r) [∂_r(r a_θ) − ∂_θ a_r]
        inv_r_cc = diag(rc, tc, lambda r, t: 1.0 / r + 0 * t)
        r_rc = diag(re, tc, lambda r, t: r + 0 * t)
        Dr_rc2cc = _curl_radial_diff(re, rc, order)  # ∂_r RC->CC
        Dt_ce2cc = _stag_theta_diff_parity(te, tc, order, par_v)  # ∂_θ a_r CE->CC
        blocks["cphi_ath"] = blk(
            inv_r_cc @ sp.kron(Dr_rc2cc, sp.eye(Nt_c)) @ r_rc, "ph", "th")
        blocks["cphi_ar"] = blk(
            -inv_r_cc @ sp.kron(sp.eye(Nr_c), Dt_ce2cc), "ph", "r")

    blocks["_src"] = src
    blocks["_dst"] = dst
    return blocks


def build_staggered_grad_div(grids: "StaggeredGrids", m, order=2):
    """Staggered grad-div operator ∇(∇·v), velocity state → velocity state,
    as the composition G ∘ Div of the staggered gradient and its norm-adjoint
    divergence. Since Div is the norm-adjoint of G the composition is
    self-adjoint and negative-semidefinite under the velocity volume norm
    W_v. This is the ⅓∇(∇·v) piece of the viscous stress divergence
    ∇·D = ∇²v + ⅓∇(∇·v) (constant ρ₀ν). ``order`` threads through the
    shared gradient builder.

    On a ball (order 2) the radial part of the divergence is the flux form
    with the face weights r_{c,i-1} r_{c,i} in place of r_i²
    (``_origin_flux_divergence``): the midpoint form is exact on v_r = 1
    but 33 % off on v_r = r in the first cell, and the gradient of that
    O(1) cell error is O(1/h) on the first face; the product form is exact
    on both at every cell and O(h) on r² in the first, at the price of the
    exact self-adjointness of this block near the origin (the pair G, Div
    of the saddle point is untouched).
    """
    G, Div, _, _ = build_staggered_grad_div_pair(grids, m, order=order)
    if order <= 2 and bool(abs(float(grids.r_edge[0])) < 1e-12):
        Div = _origin_flux_divergence(grids, Div)
    return (G @ Div).tocsr()


def _origin_flux_divergence(grids, Div):
    """``Div`` with its radial block replaced on a ball by the flux form
    (A_{i+1} v_{i+1} − A_i v_i) / (r_{c,i}² h) with the face weights
    A_i = r_{c,i-1} r_{c,i} = r_i² − h²/4 (−h²/4 at the origin): the
    two-point flux difference that is exact on v_r ∈ {1, r} in every cell,
    the origin ring entering the first cell as v_r(0)/h."""
    re, rc, tc = grids.r_edge, grids.r_center, grids.theta_center
    Nr_e, Nr_c, Nt_c = len(re), len(rc), len(tc)
    hr = re[1] - re[0]
    A = re**2 - 0.25 * hr**2
    D = np.zeros((Nr_c, Nr_e), dtype=complex)
    for i in range(Nr_c):
        D[i, i] = -A[i] / (rc[i]**2 * hr)
        D[i, i + 1] = A[i + 1] / (rc[i]**2 * hr)
    I_tc = sp.eye(Nt_c, format="csr", dtype=complex)
    Div = Div.tocsr()
    return sp.hstack([sp.kron(sp.csr_matrix(D), I_tc, format="csr"),
                      Div[:, Nr_e * Nt_c:]], format="csr")


def _assemble_curl_blocks(grids, m, order, dual, wall_bc=None):
    """Assemble the six named blocks of ``build_staggered_curl`` into one
    (3-component dst) × (3-component src) sparse matrix in component order
    (r, θ, φ); missing blocks are zero. ``wall_bc`` threads through."""
    B = build_staggered_curl(grids, m, order=order, dual=dual, wall_bc=wall_bc)
    src, dst = B["_src"], B["_dst"]
    keys = ("r", "th", "ph")
    src_n = {k: src[k][0] * src[k][1] for k in keys}
    dst_n = {k: dst[k][0] * dst[k][1] for k in keys}
    pairs = {("r", "ph"): "cr_aphi", ("r", "th"): "cr_ath",
             ("th", "r"): "cth_ar", ("th", "ph"): "cth_aphi",
             ("ph", "th"): "cphi_ath", ("ph", "r"): "cphi_ar"}
    rows = []
    for dk in keys:
        rb = []
        for sk in keys:
            key = pairs.get((dk, sk))
            if key is not None and key in B:
                rb.append(B[key])
            else:
                rb.append(sp.csr_matrix((dst_n[dk], src_n[sk]), dtype=complex))
        rows.append(sp.hstack(rb, format="csr"))
    return sp.vstack(rows, format="csr")


def _curl_output_norm(grids):
    """Diagonal volume norm W_c on the primal-curl output grids (curl_r:CE,
    curl_θ:RC, curl_φ:EE), with the same r² sinθ·hr·ht volume element as
    ``build_staggered_grad_div_pair``. Pole θ-edges carry zero weight."""
    re, rc = grids.r_edge, grids.r_center
    te, tc = grids.theta_edge, grids.theta_center
    hr = re[1] - re[0]
    ht = tc[1] - tc[0] if len(tc) > 1 else np.pi

    def vol(rv, tv):
        return ((rv**2)[:, None] * np.sin(tv)[None, :]).ravel().astype(complex) \
            * hr * ht
    return np.concatenate([vol(rc, te), vol(re, tc), vol(re, te)])


def build_staggered_curl_curl(grids: "StaggeredGrids", m, order=2,
                              wall_bc=None):
    """Staggered curl-curl operator ∇×(∇×v), velocity state → velocity state,
    as the norm-weighted adjoint composition

        curl_curl = W_v⁻¹ Cpᵀ W_c Cp

    with Cp the primal curl (velocity grids → curl grids) and W_v, W_c the
    velocity and curl-output volume norms. The Gram structure makes the
    operator self-adjoint and positive-semidefinite under W_v for any Cp
    (the differential dual-curl product Cd·Cp is equally consistent but not
    self-adjoint), and it keeps that structure under the row+column wall
    elimination of ``_eliminate_wall_vr``.

    ``wall_bc`` (None / 'stress_free' / 'noslip') selects the ghost closure
    of the wall row of ∂_r(r·v_tan) in Cp. The natural wall condition of
    the Gram form is that this wall row vanish, n×(∇×v) = 0 in the
    continuum, whatever closure the row carries; the traction-free
    (stress-free) wall is the natural condition of the strain Gram
    ``build_staggered_weighted_viscous_gram`` instead. Pole θ-edges carry
    zero weight in W_c; the 1/W_v guard zeros the weightless rows.
    """
    Cp = _assemble_curl_blocks(grids, m, order, dual=False, wall_bc=wall_bc)
    _, _, W_v, _ = build_staggered_grad_div_pair(grids, m, order=order)
    W_c = _curl_output_norm(grids)
    inv_Wv = np.where(np.abs(W_v) > 1e-30,
                      1.0 / np.where(np.abs(W_v) > 1e-30, W_v, 1.0), 0.0)
    return (sp.diags(inv_Wv) @ Cp.conj().T @ sp.diags(W_c) @ Cp).tocsr()


def build_staggered_vector_laplacian(grids: "StaggeredGrids", m, order=2,
                                     wall_bc=None):
    """Staggered vector Laplacian ∇²v = ∇(∇·v) − ∇×(∇×v), velocity state →
    velocity state. Both pieces are norm-adjoint compositions, so ∇²v is
    self-adjoint under W_v by construction.

    The norm-adjoint Div has an inconsistent radial-wall layer (see
    ``build_staggered_grad_div_pair``), so the assembled operator is correct
    in the radial interior; applying ``_eliminate_wall_vr`` (v_r = 0 as
    row+column elimination) yields the operator on the constrained space,
    consistent to 2nd order including the wall-adjacent rows and still
    self-adjoint. ``wall_bc`` threads into the curl-curl piece, whose
    natural wall condition is n×(∇×v) = 0 (see
    ``build_staggered_curl_curl``); the stress-free viscous force is the
    unit-density strain Gram.
    """
    GD = build_staggered_grad_div(grids, m, order=order)
    CC = build_staggered_curl_curl(grids, m, order=order, wall_bc=wall_bc)
    return (GD - CC).tocsr()


# ---------------------------------------------------------------------------
# Strain-rate operators and the density-weighted viscous force
# ---------------------------------------------------------------------------

def build_staggered_strain(grids, m, order=2, wall_bc=None, pole_closure=False):
    """Discrete rate-of-strain operator e = ½(∇v + (∇v)ᵀ) in physical
    components, mapping the velocity state [v_r; v_θ; v_φ] to the six
    independent symmetric-tensor components on their natural C-grid
    locations:

        e_rr, e_θθ, e_φφ  : CC (r_center, θ_center)
        e_rθ              : EE (r_edge,   θ_edge)
        e_rφ              : RC (r_edge,   θ_center)
        e_θφ              : CE (r_center, θ_edge)

    with (∂_φ → im)
        e_rr = ∂_r v_r
        e_θθ = (1/r)∂_θ v_θ + v_r/r
        e_φφ = (im/(r sinθ)) v_φ + v_r/r + (cotθ/r) v_θ
        e_rθ = ½[ r ∂_r(v_θ/r) + (1/r)∂_θ v_r ]
        e_rφ = ½[ (im/(r sinθ)) v_r + r ∂_r(v_φ/r) ]
        e_θφ = ½[ (1/r)∂_θ v_φ − (cotθ/r) v_φ + (im/(r sinθ)) v_θ ]
    and trace e_rr + e_θθ + e_φφ = ∇·v.

    θ-derivatives use the source field's pole parity (v_r: (-1)^|m|; v_θ,
    v_φ: (-1)^|m+1|). 1/sinθ, cotθ and (on a ball) 1/r are zero-guarded at
    the poles / origin edge, which carry zero norm weight. The radial
    derivative of the divided tangential fields g = v_tan/r (in e_rθ, e_rφ)
    has its own wall closure: with ``wall_bc='stress_free'``
    (∂_r(v_tan/r) = 0) the wall-edge rows are zeroed (even reflection of g);
    with ``'noslip'`` (v_tan = 0 on the wall) the mirror ghost v_g = −v_0
    gives g_g = −(r_0/r_g) g_0 and the wall row (g_0 − g_g)/(r_0 − r_g);
    with None the plain one-sided row is kept.

    On a ball r ∂_r(v_tan/r) is spelled ∂_r v_tan − v_tan/r with the plain
    two-point derivative (vector origin parity on the weightless origin
    row) and the two-point face average: the two-point difference of g is
    exact on v_tan ∈ {r, r², r³} but 33 % off on the constant part of a
    field finite at the origin (v_tan(0) = c at m = 0, |m| = 1), whose
    g = c/r it differences across the first face, and the shear it leaves
    there is O(1/h); the spelling here is the two-point stencil exact on
    v_tan ∈ {1, r}, second order on the rest. The no-slip ghost enters both
    the derivative (−2 v_0 / h) and the average (0). The two-point average
    of v_r over the first cell keeps the O(h) error of v_r/r there: it is
    the weight that makes the transposed form, −(σ_θθ + σ_φφ)/r on the
    first faces, exact on a constant stress.

    ``pole_closure`` (order 2) gives the pole rows of e_θφ, otherwise
    zero, the parity limit of a regular field: the two v_φ terms cancel
    there and (im/(r sinθ)) v_θ → (im/r) v_θ'(0) = (im/(2 r Δθ)) (1 − p)
    v_θ(adjacent edge) with p = (−1)^|m+1|, nonzero at even |m| ≥ 2 only.
    The pole edges stay weightless; ``_strain_pole_share`` is the piece
    of the norm-adjoint divergence that reads them.

    Returns (S, comp_grids): S stacks the six component blocks (row order
    rr, θθ, φφ, rθ, rφ, θφ) and comp_grids lists (n_rows, (r_nodes, θ_nodes))
    per component for building the tensor volume norm.
    """
    re, rc = grids.r_edge, grids.r_center
    te, tc = grids.theta_edge, grids.theta_center
    Nr_e, Nr_c = len(re), len(rc)
    Nt_e, Nt_c = len(te), len(tc)
    hr = re[1] - re[0]
    ht = tc[1] - tc[0] if Nt_c > 1 else np.pi
    im = 1j * m
    par_s = (-1) ** (abs(m) % 2)          # v_r / scalar parity
    par_v = (-1) ** ((abs(m) + 1) % 2)    # v_θ, v_φ parity

    I_tc = sp.eye(Nt_c, format="csr", dtype=complex)
    I_te = sp.eye(Nt_e, format="csr", dtype=complex)
    I_rc = sp.eye(Nr_c, format="csr", dtype=complex)
    I_re = sp.eye(Nr_e, format="csr", dtype=complex)

    def dgrid(rv, tv, fn):
        return sp.diags(np.asarray(fn(rv[:, None], tv[None, :])).ravel()
                        .astype(complex), format="csr")

    def inv_sin_guard(t):
        s = np.sin(t)
        nz = np.abs(s) > 1e-14
        return np.where(nz, 1.0 / np.where(nz, s, 1.0), 0.0)

    def cot_guard(t):
        s = np.sin(t)
        nz = np.abs(s) > 1e-14
        return np.where(nz, np.cos(t) / np.where(nz, s, 1.0), 0.0)

    def inv_r_guard(r):
        # 1/r with a zero at the weightless ball origin edge
        r = np.asarray(r, float)
        nz = np.abs(r) > 1e-14
        return np.where(nz, 1.0 / np.where(nz, r, 1.0), 0.0)

    # 1-D radial first derivatives centers→edges and edges→centers
    if order <= 2:
        Dr_c2e = _stag_diff_1d(Nr_e, Nr_c, hr)
        Dr_e2c = _stag_diff_e2c(Nr_c, hr)
    else:
        Dr_c2e = _stag_diff_1d_lagrange(rc, re, order)
        Dr_e2c = _stag_diff_1d_lagrange(re, rc, order)

    # centers→edges derivative of the divided tangential fields g = v_tan/r
    # (wall closure by wall_bc); on a ball the same term is spelled
    # ∂_r v_tan − v_tan/r with the plain derivative (vector parity at the
    # origin row) and the face average, see below
    is_ball_s = bool(abs(float(re[0])) < 1e-12)
    op_vec_s = _origin_parity(m, "vector")
    wall_bc = normalize_wall_bc(wall_bc)
    if is_ball_s:
        Dr_c2e_tan = _stag_diff_1d_origin_parity(rc, re, max(order, 2), op_vec_s)
        Pr_c2e_tan = (_avg_1d(Nr_e, Nr_c) if order <= 2
                      else _stag_interp_1d_lagrange(rc, re, order))
    else:
        Dr_c2e_tan = Dr_c2e
        Pr_c2e_tan = None
    walls = [Nr_e - 1] if is_ball_s else [0, Nr_e - 1]
    if wall_bc is not None:
        # the wall rows are closed per theta node below
        Dr_c2e_tan = Dr_c2e_tan.tolil()
        if is_ball_s:
            Pr_c2e_tan = Pr_c2e_tan.tolil()
        for i in walls:
            Dr_c2e_tan.rows[i], Dr_c2e_tan.data[i] = [], []
            if is_ball_s:
                Pr_c2e_tan.rows[i], Pr_c2e_tan.data[i] = [], []
        Dr_c2e_tan = Dr_c2e_tan.tocsr()
        if is_ball_s:
            Pr_c2e_tan = Pr_c2e_tan.tocsr()

    def close_walls(D2, P2, kind, n_t):
        """The wall rows of the stacked radial derivative D2 (and, on a
        ball, the average P2) of a tangential component per theta node:
        the slip ghost of ``wall_bc``, the mirror ghost v_g = rho v_0 on
        the ball spelling d_r v_tan - v_tan / r (derivative (1 - rho) /
        (r0 - rg), average (1 + rho) / 2) or g_g = gamma g_0 on the shell
        spelling r d_r(v_tan / r) (derivative (1 - gamma) / (r0 - rg));
        ls = 0 is the no-slip ghost, ls = inf the even reflection of g,
        whose rows are empty."""
        if wall_bc is None:
            return D2, P2
        D2 = D2.tolil()
        if P2 is not None:
            P2 = P2.tolil()
        for i in walls:
            j0 = Nr_c - 1 if i == Nr_e - 1 else 0
            r0, rw = rc[j0], re[i]
            rg = 2.0 * rw - r0
            ls = wall_bc.lengths(1 if i == Nr_e - 1 else 0, kind, n_t)
            if is_ball_s:
                rho = _slip_ratio(ls, rw, r0, hr)
            else:
                rho = _slip_ratio_divided(ls, rw, r0, hr)
            for j in range(n_t):
                row, col = i * n_t + j, j0 * n_t + j
                if np.isinf(ls[j]):
                    continue
                D2[row, col] = complex((1.0 - rho[j]) / (r0 - rg))
                if P2 is not None:
                    avg = (1.0 + rho[j]) / 2.0
                    if avg != 0.0:
                        P2[row, col] = complex(avg)
        D2 = D2.tocsr()
        D2.sort_indices()
        if P2 is not None:
            P2 = P2.tocsr()
            P2.sort_indices()
        return D2, P2

    def open_wall_factor(M, kind, n_t):
        """The v_r part of a wall shear on an open wall (wall v_r faces
        kept) scaled by ``_slip_wall_shear_factor``; untouched on a closed
        wall, where the wall v_r vanishes."""
        if wall_bc is None or not any(wall_bc.open):
            return M
        M = M.tolil()
        for i in walls:
            k = 1 if i == Nr_e - 1 else 0
            if not wall_bc.open[k]:
                continue
            j0 = Nr_c - 1 if i == Nr_e - 1 else 0
            r0, rw = rc[j0], re[i]
            f = _slip_wall_shear_factor(wall_bc.lengths(k, kind, n_t), rw, r0, hr,
                                        not is_ball_s)
            for j in range(n_t):
                row = i * n_t + j
                M.data[row] = [complex(v * f[j]) for v in M.data[row]]
        M = M.tocsr()
        M.eliminate_zeros()
        return M

    # radial interpolation edges→centers and the 1-D θ stencils (the
    # 2-point centers→edges stencils have zero pole rows, which the
    # weightless pole norm makes inert; parity-extended above order 2)
    if order <= 2:
        Pr_e2c = _avg_e2c(Nr_c)
        Dt_e2c = _stag_diff_e2c(Nt_c, ht)
        Dt_c2e = _stag_diff_c2e_zero_ends(Nt_e, ht)
        Pt_e2c = _avg_e2c(Nt_c)
        Pt_c2e = _avg_c2e_zero_ends(Nt_e)
    else:
        Pr_e2c = _stag_interp_1d_lagrange(re, rc, order)
        Dt_e2c = _stag_theta_diff_parity(te, tc, order, par_v)
        Dt_c2e = _stag_theta_diff_parity(tc, te, order, par_s)
        Pt_e2c = _stag_interp_1d_lagrange(te, tc, order)
        Pt_c2e = _stag_interp_1d_lagrange(tc, te, order)

    n_vr, n_vt, n_vp = Nr_e * Nt_c, Nr_c * Nt_e, Nr_c * Nt_c
    Z = lambda nr, nc: sp.csr_matrix((nr, nc), dtype=complex)

    # e_rr = ∂_r v_r on CC (∂_r re→rc, θ_center shared)
    n_cc = Nr_c * Nt_c
    e_rr_vr = sp.kron(Dr_e2c, I_tc, format="csr")             # CC←RC
    e_rr = sp.hstack([e_rr_vr, Z(n_cc, n_vt), Z(n_cc, n_vp)], format="csr")

    # e_θθ = (1/r)∂_θ v_θ + v_r/r on CC
    inv_r_cc = dgrid(rc, tc, lambda r, t: 1.0 / r + 0 * t)
    eth_vt = inv_r_cc @ sp.kron(I_rc, Dt_e2c, format="csr")   # CC←CE
    eth_vr = inv_r_cc @ sp.kron(Pr_e2c, I_tc, format="csr")   # CC←RC
    e_thth = sp.hstack([eth_vr, eth_vt, Z(n_cc, n_vp)], format="csr")

    # e_φφ = (im/(r sinθ)) v_φ + v_r/r + (cotθ/r) v_θ on CC
    coef_imrs_cc = dgrid(rc, tc, lambda r, t: im * inv_sin_guard(t) / r)
    eph_vp = coef_imrs_cc                                     # v_φ already CC
    eph_vr = inv_r_cc @ sp.kron(Pr_e2c, I_tc, format="csr")   # v_r/r  CC←RC
    cot_over_r_cc = dgrid(rc, tc, lambda r, t: cot_guard(t) / r)
    eph_vt = cot_over_r_cc @ sp.kron(I_rc, Pt_e2c, format="csr")  # CC←CE (θ interp)
    e_phph = sp.hstack([eph_vr, eph_vt, eph_vp], format="csr")

    # e_rθ = ½[ r ∂_r(v_θ/r) + (1/r)∂_θ v_r ] on EE
    n_ee = Nr_e * Nt_e
    inv_r_ce = dgrid(rc, te, lambda r, t: 1.0 / r + 0 * t)
    r_ee = dgrid(re, te, lambda r, t: r + 0 * t)
    inv_r_ee = dgrid(re, te, lambda r, t: inv_r_guard(r) + 0 * t)
    D_te, P_te = close_walls(sp.kron(Dr_c2e_tan, I_te, format="csr"),
                             None if Pr_c2e_tan is None
                             else sp.kron(Pr_c2e_tan, I_te, format="csr"), "edges", Nt_e)
    if is_ball_s:
        erth_vt = 0.5 * (D_te - inv_r_ee @ P_te)                              # EE←CE
    else:
        erth_vt = 0.5 * r_ee @ D_te @ inv_r_ce                                # EE←CE
    erth_vr = 0.5 * inv_r_ee @ sp.kron(I_re, Dt_c2e, format="csr")         # EE←RC
    erth_vr = open_wall_factor(erth_vr, "edges", Nt_e)
    e_rth = sp.hstack([erth_vr, erth_vt, Z(n_ee, n_vp)], format="csr")

    # e_rφ = ½[ (im/(r sinθ)) v_r + r ∂_r(v_φ/r) ] on RC
    coef_imrs_rc = dgrid(re, tc, lambda r, t: im * inv_sin_guard(t) * inv_r_guard(r))
    erph_vr = open_wall_factor(0.5 * coef_imrs_rc, "centers", Nt_c)   # v_r already RC
    inv_r_cc2 = dgrid(rc, tc, lambda r, t: 1.0 / r + 0 * t)
    r_rc = dgrid(re, tc, lambda r, t: r + 0 * t)
    D_tc, P_tc = close_walls(sp.kron(Dr_c2e_tan, I_tc, format="csr"),
                             None if Pr_c2e_tan is None
                             else sp.kron(Pr_c2e_tan, I_tc, format="csr"), "centers", Nt_c)
    if is_ball_s:
        inv_r_rc = dgrid(re, tc, lambda r, t: inv_r_guard(r) + 0 * t)
        erph_vp = 0.5 * (D_tc - inv_r_rc @ P_tc)                              # RC←CC
    else:
        erph_vp = 0.5 * r_rc @ D_tc @ inv_r_cc2                               # RC←CC
    n_rc = Nr_e * Nt_c
    e_rph = sp.hstack([erph_vr, Z(n_rc, n_vt), erph_vp], format="csr")

    # e_θφ = ½[ (1/r)∂_θ v_φ − (cotθ/r) v_φ + (im/(r sinθ)) v_θ ] on CE
    inv_r_ce2 = dgrid(rc, te, lambda r, t: 1.0 / r + 0 * t)
    ethph_vp_d = inv_r_ce2 @ sp.kron(I_rc, Dt_c2e, format="csr")           # CE←CC
    cot_over_r_ce = dgrid(rc, te, lambda r, t: cot_guard(t) / r)
    ethph_vp_a = -cot_over_r_ce @ sp.kron(I_rc, Pt_c2e, format="csr")      # CE←CC
    ethph_vp = 0.5 * (ethph_vp_d + ethph_vp_a)
    coef_imrs_ce = dgrid(rc, te, lambda r, t: im * inv_sin_guard(t) / r)
    ethph_vt = 0.5 * coef_imrs_ce
    if pole_closure:
        pole = sp.lil_matrix((Nt_e, Nt_e), dtype=complex)
        pole[0, 1] = pole[Nt_e - 1, Nt_e - 2] = (1 - par_v) / (2.0 * ht)
        ethph_vt = ethph_vt + 0.5 * im * (inv_r_ce2 @ sp.kron(I_rc, pole.tocsr(), format="csr"))
    n_ce = Nr_c * Nt_e
    e_thph = sp.hstack([Z(n_ce, n_vr), ethph_vt, ethph_vp], format="csr")

    S = sp.vstack([e_rr, e_thth, e_phph, e_rth, e_rph, e_thph], format="csr")
    comp_grids = [
        (n_cc, (rc, tc)), (n_cc, (rc, tc)), (n_cc, (rc, tc)),
        (n_ee, (re, te)), (Nr_e * Nt_c, (re, tc)), (n_ce, (rc, te)),
    ]
    return S, comp_grids


def _pole_vtheta_prolongation(grids):
    """Prolongation Q (n_v × n_v) on the velocity state that replaces the
    weightless pole θ-edge v_θ DOFs by the adjacent interior θ-edge value
    (the even extension of a field finite at the pole), identity elsewhere
    and zero on the pole columns. Used at |m| = 1, where v_θ and v_φ are
    finite at the poles and the strain rows at the first θ-center need the
    pole value for their (i m v_φ + cosθ v_θ)/sinθ and ∂_θ v_θ terms; for
    every other m the pole value is zero and the DOFs are inert."""
    Nr_e, Nr_c = len(grids.r_edge), len(grids.r_center)
    Nt_e, Nt_c = len(grids.theta_edge), len(grids.theta_center)
    n_vr, n_vt, n_vp = Nr_e * Nt_c, Nr_c * Nt_e, Nr_c * Nt_c
    Q = sp.lil_matrix((n_vr + n_vt + n_vp, n_vr + n_vt + n_vp), dtype=complex)
    Q.setdiag(1.0)
    for ir in range(Nr_c):
        base = n_vr + ir * Nt_e
        for pole, inner in ((0, 1), (Nt_e - 1, Nt_e - 2)):
            Q[base + pole, base + pole] = 0.0
            Q[base + pole, base + inner] = 1.0
    return Q.tocsr()


def _strain_pole_share(grids, comps, name):
    """The share of the pole θ-edges in the norm-adjoint divergence of a
    tensor placement, (n_v × n_T) on the stacked components ``comps``
    (names in placement order, on their C-grid locations), reading the
    component ``name`` ('tp' on the six, 'pt' on the nine).

    (∇·T)_φ carries (cotθ/r) T_φθ, whose adjoint of the centres-to-edges
    average is the sinθ-weighted trapezoid over a cell's two edges,
    (1/(2 r sinθ_c)) Σ_e cosθ_e T(e); at the pole-adjacent cell the pole
    edge has zero weight and its share, T(pole) / (2 r sinθ_c) with
    cosθ = ±1, is lost although cotθ W_e → r² Δr Δθ is finite there: an
    O(1/Δθ) truncation on the pole-adjacent v_φ ring wherever T_φθ has a
    finite pole limit (even |m| ≥ 2 for the strain, every m for a
    non-symmetric tensor). The share is added explicitly, reading the
    pole DOF (the parity limit carried by the map's pole row); it is not
    the norm adjoint of anything, so W F loses Hermiticity on that ring,
    by the term the adjoint cannot represent (its transpose would feed
    the pole limit back into the v_θ ring, whose lumped weight already
    integrates the hat function exactly)."""
    re, rc, te, tc = grids.r_edge, grids.r_center, grids.theta_edge, grids.theta_center
    Nr_e, Nr_c, Nt_e, Nt_c = len(re), len(rc), len(te), len(tc)
    sizes = {"CC": Nr_c * Nt_c, "EE": Nr_e * Nt_e, "RC": Nr_e * Nt_c, "CE": Nr_c * Nt_e}
    locs = {"rr": "CC", "tt": "CC", "pp": "CC", "rt": "EE", "tr": "EE",
            "rp": "RC", "pr": "RC", "tp": "CE", "pt": "CE"}
    off = 0
    for c in comps:
        if c == name:
            break
        off += sizes[locs[c]]
    n_T = sum(sizes[locs[c]] for c in comps)
    n_v = sizes["RC"] + sizes["CE"] + sizes["CC"]
    vp0 = sizes["RC"] + sizes["CE"]
    share = sp.lil_matrix((n_v, n_T), dtype=complex)
    for ir in range(Nr_c):
        w = 1.0 / (2.0 * rc[ir] * np.sin(tc[0]))
        share[vp0 + ir * Nt_c, off + ir * Nt_e] = w
        share[vp0 + ir * Nt_c + Nt_c - 1, off + ir * Nt_e + Nt_e - 1] = -w
    return share.tocsr()


def build_staggered_weighted_viscous_gram(grids, m, rho_fn, order=2,
                                          wall_bc=None, pole_closure=False):
    """Density-weighted viscous force

        F v = (1/ρ₀) ∇·(ρ₀ T),   T = 2e − (2/3)(∇·v) I,   e = ½(∇v+(∇v)ᵀ)

    built as the strain Gram form

        F = −(1/ρ₀) W_v⁻¹ Sᵀ (ρ₀ W_T) C S,   C = 2𝕀 − (2/3) 𝟙_diag𝟙_diagᵀ

    where S = ``build_staggered_strain`` (velocity → 6 tensor components),
    W_T is the per-component tensor volume norm (off-diagonal components
    counted twice from the contraction T:e; pole θ-edges weightless), and C
    maps strain to deviatoric stress (the trace couples the three diagonal
    components, which share the CC grid, so C commutes with ρ₀W_T).

    F is self-adjoint and negative-semidefinite under the ρ₀-weighted
    velocity norm ρ₀W_v on the weightful subspace (pole θ-edge velocity DOFs
    carry zero norm and are inert), and reduces to the unweighted
    ∇²v + ⅓∇(∇·v) to discretization order for constant ρ₀. It stays
    self-adjoint under the downstream row+column elimination of the wall
    v_r DOFs. On a ball, 1/W_v is additionally zeroed at the pole θ-edges
    of v_θ (W_v ∝ r² sinθ is doubly vanishing there and would inject huge
    entries); on a shell only the |W_v| > 1e-30 guard applies.

    At |m| = 1 the strain acts on S·Q with Q = ``_pole_vtheta_prolongation``:
    the tangential components are finite at the poles, and the weightless
    pole θ-edge v_θ DOFs (zero in every eigenmode) must read the adjacent
    interior value or the first θ-center's strain is off by O(1/Δθ).

    ``rho_fn(r, θ)`` accepts broadcast 2-D arrays. ``order`` and ``wall_bc``
    thread into ``build_staggered_strain``. ``pole_closure`` adds the pole
    rows of the strain and the pole share of the divergence
    (``_strain_pole_share``): the pole-adjacent v_φ rows then read
    ρ₀ σ_θφ(pole) / (2 r sinθ_c), which makes them consistent at even
    |m| ≥ 2 at the price of the exact self-adjointness on that ring.
    """
    re, rc = grids.r_edge, grids.r_center
    te, tc = grids.theta_edge, grids.theta_center
    S, WT, inv_Wv, comp_grids = _strain_operator_and_norms(grids, m, order, wall_bc,
                                                           pole_closure)

    def rho_on(rg, tg):
        return np.asarray(rho_fn(rg[:, None] + 0 * tg[None, :],
                                 tg[None, :] + 0 * rg[:, None])).ravel().astype(complex)

    rhoT = np.concatenate([rho_on(rg, tg) for _, (rg, tg) in comp_grids])
    rho_v = np.concatenate([rho_on(re, tc), rho_on(rc, te), rho_on(rc, tc)])
    C = _deviatoric_map(comp_grids)
    K = (S.conj().T @ sp.diags(rhoT * WT) @ C @ S).tocsr()
    F = -(sp.diags(1.0 / rho_v) @ sp.diags(inv_Wv) @ K).tocsr()
    if pole_closure:
        share = _strain_pole_share(grids, ("rr", "tt", "pp", "rt", "rp", "tp"), "tp")
        F = F + sp.diags(1.0 / rho_v) @ share @ sp.diags(rhoT) @ C @ S
    return F.tocsr()


def _strain_operator_and_norms(grids, m, order, wall_bc, pole_closure=False):
    """(S, W_T, inv_W_v, comp_grids) of the strain Gram: the strain map of
    ``build_staggered_strain`` (times the pole prolongation at |m| = 1),
    the six-component tensor norm (off-diagonals counted twice; pole
    θ-edges weightless; the no-slip wall shear faces at half weight) and
    the guarded inverse velocity norm (weightless pole v_θ DOFs dropped,
    the pole rows too on a ball; the ball origin ring at the weight of
    ``_origin_ring_volume``, so its rows carry the first cell's momentum
    balance rather than nothing)."""
    re, rc = grids.r_edge, grids.r_center
    te, tc = grids.theta_edge, grids.theta_center
    hr = re[1] - re[0]
    ht = tc[1] - tc[0] if len(tc) > 1 else np.pi
    S, comp_grids = build_staggered_strain(grids, m, order=order,
                                           wall_bc=wall_bc, pole_closure=pole_closure)
    if abs(int(m)) == 1:
        S = (S @ _pole_vtheta_prolongation(grids)).tocsr()
    mult = [1, 1, 1, 2, 2, 2]   # off-diagonals counted twice in T:e
    is_ball = bool(np.isclose(re[0], 0.0))
    walls = [len(re) - 1] if is_ball else [0, len(re) - 1]
    wall_bc = normalize_wall_bc(wall_bc)
    WT = []
    for k, ((_, (rg, tg)), mu) in enumerate(zip(comp_grids, mult)):
        w = (rg[:, None]**2 * np.sin(tg)[None, :]).ravel() * hr * ht
        pole = np.abs((np.sin(tg)[None, :] + 0 * rg[:, None]).ravel()) < 1e-12
        w = np.where(pole, 0.0, w)
        if wall_bc is not None and k in (3, 4):
            # the wall shear faces at the weight of the slip closure
            w = w.reshape(len(rg), len(tg))
            for i in walls:
                ls = wall_bc.lengths(1 if i == len(re) - 1 else 0,
                                     "edges" if k == 3 else "centers", len(tg))
                w[i, :] *= _slip_face_weight(ls, hr)
            w = w.ravel()
        WT.append(mu * w.astype(complex))
    WT = np.concatenate(WT)
    _, _, W_v, _ = build_staggered_grad_div_pair(grids, m, order=order)
    if wall_bc is not None and any(wall_bc.open):
        # an open wall's v_r faces carry the half cell inside the wall
        W_v = W_v.copy()
        for i in walls:
            if wall_bc.open[1 if i == len(re) - 1 else 0]:
                W_v[i * len(tc):(i + 1) * len(tc)] *= 0.5
    base_keep = np.abs(W_v) > 1e-12 * np.max(np.abs(W_v))
    if is_ball:
        Nt_e_ = len(te)
        n_vr_ = len(re) * len(tc)
        pole_mask = np.zeros(len(W_v), dtype=bool)
        for ir in range(len(rc)):
            pole_mask[n_vr_ + ir * Nt_e_ + 0] = True
            pole_mask[n_vr_ + ir * Nt_e_ + (Nt_e_ - 1)] = True
        base_keep = base_keep & (~pole_mask)
        W_v = W_v.copy()
        W_v[:len(tc)] = _origin_ring_volume(grids)
        base_keep[:len(tc)] = True
    inv_Wv = np.where(base_keep, 1.0 / np.where(base_keep, W_v, 1.0), 0.0)
    return S, WT, inv_Wv, comp_grids


def _origin_ring_volume(grids):
    """Volume weight (h³/15) sinθ_j Δθ of the origin ring v_r(0, θ_j) on a
    ball, the inertia of the ring rows of the strain Gram.

    The face has zero area, so no flux crosses it and the norm adjoints
    of the operator library never read the ring; the weight only turns
    the Gram's ring rows, otherwise guarded to zero, into the momentum
    balance of the first cell, which the fold of ``_reduce_velocity_dofs``
    projects onto the Cartesian origin amplitude a with the l = 1 profile
    p_j (sinθ_j at |m| = 1, cosθ_j at m = 0). The ring basis function
    strains the first cell only, e = (−1, 1, 1) p_j / h on its diagonal,
    so with the trace-free stress σ the fold of K = Sᵀ W_T C S is
    Σ_j p_j (K u)_j = −(h²/2) Δθ Σ_j sinθ_j p_j σ_rr(h/2, θ_j). The origin
    value of σ_rr has no l = 1 content; the radial derivative of the
    exact stress integrates to (8/15) a_{∇·σ} against sin²θ, which would
    make the fold −(2h³/15) a_{∇·σ}, and the discrete strain adds the
    first-cell error of v_r/r (the two-point average over r_c = h/2,
    (h/4) ∂_r² v_r in e_θθ and e_φφ), whose fold is
    (2h³/45) a_{∇²u + 2∇(∇·u)}: on a solenoidal field the two make
    −(4h³/45) a_{∇·σ}. The mass of the fold is Σ_j W_j p_j² = (4/3) w₀,
    which balances it at w₀ = h³/15 (measured 0.063 h³ on the manufactured
    |m| = 1 field, where ∇·u ≠ 0 and the two-point θ stencil leaves an
    O(Δθ²/h) a reaction term in the first cell's e_θθ); any O(h³) weight
    leaves the master row consistent to O(1), an O(h²) error on the
    ring's O(h³) volume, but with the wrong inertia."""
    re, tc = grids.r_edge, grids.theta_center
    hr = re[1] - re[0]
    ht = tc[1] - tc[0] if len(tc) > 1 else np.pi
    return (hr**3 / 15.0 * np.sin(tc) * ht).astype(complex)


def _deviatoric_map(comp_grids):
    """C = 2I - (2/3) Tr on the six strain components: strain to
    deviatoric stress, the trace coupling the three CC diagonals."""
    n_cc = comp_grids[0][0]
    offs = np.cumsum([0] + [c[0] for c in comp_grids])
    n_T = int(offs[-1])
    C = sp.eye(n_T, format="csr", dtype=complex) * 2.0
    Tr = sp.lil_matrix((n_T, n_T), dtype=complex)
    eye_cc = sp.eye(n_cc, format="csr", dtype=complex)
    for a in range(3):
        for b in range(3):
            Tr[offs[a]:offs[a] + n_cc, offs[b]:offs[b] + n_cc] += eye_cc
    return (C - (2.0 / 3.0) * Tr.tocsr()).tocsr()


# ---------------------------------------------------------------------------
# Wall and origin DOF reduction
# ---------------------------------------------------------------------------

def _reduce_velocity_dofs(A, B, layout, grids, m, vel_name, extra_drop=(),
                          walls=True, origin="regular"):
    """Symmetric wall/origin DOF reduction A_red = Pᴴ A P, B_red = Pᴴ B P of a
    pencil whose v_r block ``f"{vel_name}_r"`` sits at offset 0.

    The prolongation P drops the outer-wall v_r ring (v_r = 0), the
    inner-wall ring on a shell, the whole origin ring for |m| ≥ 2 on a ball
    (a regular vector has v_r(0, θ) = 0), and ``extra_drop`` (b_r wall
    DOFs); for m = 0 / |m| = 1 it folds the origin ring into its (ir=0, j=0)
    master amplitude with the cosθ / sinθ profile ratios, a Galerkin
    projection of the Nθ origin rows onto the single Cartesian DOF: with
    the ring rows of A at the weight of ``_origin_ring_volume`` the master
    row is the profile-weighted average of the ring's momentum equations
    and its mass the same average of the ring's mass. The row+column
    reduction keeps self-adjoint operators self-adjoint on the constrained
    space. ``walls=False`` keeps the wall rings (a vector potential);
    ``origin='drop'`` drops the origin ring at every m (a flux unknown,
    whose origin faces have zero area). Returns (A_red, B_red, layout)
    with the field sizes re-packed.
    """
    r_edge, th_center = grids.r_edge, grids.theta_center
    Nr_e, Nt_c = len(r_edge), len(th_center)
    is_ball = bool(np.isclose(r_edge[0], 0.0))
    n_tot = layout.total_size
    A = A.tocsr()
    B = B.tocsr()
    drop = []                     # v_r columns removed entirely
    folds = []                    # (master, slave, ratio) for m=0/1 origin
    if walls:
        # outer wall (all geometries): v_r = 0
        drop += [(Nr_e - 1) * Nt_c + it for it in range(Nt_c)]
    if not is_ball:
        if walls:
            # shell inner wall: v_r = 0
            drop += [it for it in range(Nt_c)]
    elif abs(m) >= 2 or origin == "drop":
        # ball origin, |m|≥2 or a flux: v_r(0,θ)=0 ∀θ
        drop += [it for it in range(Nt_c)]
    else:
        # ball origin, m=0/|m|=1: fold the ring into the anchor (j=0)
        profile = np.cos(th_center) if m == 0 else np.sin(th_center)
        p0 = profile[0]
        master = 0                          # (ir=0, j=0) origin v_r anchor
        for it in range(1, Nt_c):
            slave = it
            folds.append((master, slave, complex(profile[it] / p0)))
            drop.append(slave)              # slave column removed after fold
    drop += list(extra_drop)
    # prolongation P (n_tot × n_reduced)
    drop_set = set(drop)
    keep_cols = [c for c in range(n_tot) if c not in drop_set]
    col_of = {full: red for red, full in enumerate(keep_cols)}
    n_red = len(keep_cols)
    rows_P, cols_P, vals_P = [], [], []
    for red, full in enumerate(keep_cols):
        rows_P.append(full); cols_P.append(red); vals_P.append(1.0 + 0j)
    for master, slave, ratio in folds:
        # full row `slave` ← ratio · (reduced master column)
        rows_P.append(slave); cols_P.append(col_of[master]); vals_P.append(ratio)
    P = sp.csr_matrix((vals_P, (rows_P, cols_P)),
                      shape=(n_tot, n_red), dtype=complex)
    A = (P.conj().T @ A @ P).tocsr()
    B = (P.conj().T @ B @ P).tocsr()
    # re-pack the field sizes and offsets
    new_fields = []
    off = 0
    for f in layout.fields:
        sz = f.size - sum(1 for c in drop_set if f.offset <= c < f.offset + f.size)
        new_fields.append(StaggeredField(f.name, f.location, off, sz))
        off += sz
    layout = StaggeredLayout(fields=tuple(new_fields), total_size=off)
    return A, B, layout


def _apply_ball_origin_vr(A, B, m, th_center, Nt_c):
    """Origin regularity rows for v_r on a ball at the origin r-edge (ir=0,
    rows 0..Nt_c-1) of LIL matrices A, B.

    v_r(0, θ) = c·r̂(θ) for a single Cartesian origin vector c:
      * |m| ≥ 2: no m-content ⇒ v_r(0,θ) = 0 ∀θ ⇒ Dirichlet-zero rows.
      * m = 0: v_r(0,θ) = c_z cos θ ⇒ one DOF; rows j > 0 become
              v_r(0,θ_j) − (cos θ_j / cos θ_0) v_r(0,θ_0) = 0.
      * |m| = 1: v_r(0,θ) ∝ sin θ ⇒ one DOF; rows j > 0 become
              v_r(0,θ_j) − (sin θ_j / sin θ_0) v_r(0,θ_0) = 0.
    For m = 0 / |m| = 1 the j = 0 row keeps its assembled equation, so the
    eigenproblem fixes the anchor amplitude. The constraint rows carry no
    mass (B row emptied).
    """
    def _dirichlet_zero(row):
        A.rows[row] = [row]
        A.data[row] = [1.0 + 0j]
        B.rows[row] = []
        B.data[row] = []

    if abs(m) >= 2:
        for it in range(Nt_c):
            _dirichlet_zero(0 * Nt_c + it)
        return

    if m == 0:
        profile = np.cos(th_center)
    else:  # |m| == 1
        profile = np.sin(th_center)
    p0 = profile[0]
    anchor_row = 0  # (ir=0, j=0)
    for it in range(1, Nt_c):
        row = 0 * Nt_c + it
        ratio = profile[it] / p0
        A.rows[row] = [row, anchor_row]
        A.data[row] = [1.0 + 0j, -ratio + 0j]
        B.rows[row] = []
        B.data[row] = []
