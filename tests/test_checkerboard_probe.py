"""Checkerboard probe — primitive (velocity-pressure) inertial waves.

A deliberately minimal testbed for studying the colocated-grid
pressure/velocity LBB checkerboard instability, with an EXACT analytic
spectrum to measure against.

# Findings

Systematic 2×2 study (grid: colocated vs radially-staggered; divergence:
independent vs discrete-adjoint Div = -W⁻¹GᵀW) on this testbed, scored
by (a) eigenvector node-to-node alternation [checkerboard] and (b) the
spurious |Re(λ)| of modes that physically must be undamped [the inertial
spectrum is purely imaginary]:

    grid         Div            checkerboard   spurious damping
    colocated    independent    PRESENT        PRESENT
    colocated    adjoint        PRESENT        ABSENT (~1e-16)
    staggered    independent    ABSENT         PRESENT (~5e-3)
    staggered    adjoint        PRESENT        ABSENT (~1e-16)

The two pathologies have INDEPENDENT root causes:
  * Checkerboard ⟸ grid COLOCATION. Only radial staggering removes it;
    adjointness has no effect.
  * Spurious damping ⟸ NON-ADJOINT Grad/Div pair (broken discrete
    skewness ⇒ eigenvalues leave the imaginary axis). Only enforcing
    Div = -W⁻¹GᵀW removes it; staggering has no effect.

The two remedies interact: on a staggered grid, setting Div = -Gᵀ
reintroduces the checkerboard because Gᵀ inherits G's sawtooth
near-null mode (smallest-SV input mode sawtooth-content jumps 0.04→0.47).
The operator pair must therefore SIMULTANEOUSLY be (1) staggered,
(2) discretely adjoint, and (3) individually nullspace-clean so that the
adjoint transpose does not smuggle the sawtooth back in.

A clean SBP RADIAL pair (forward-diff Div, backward-diff Grad,
half-weight end norms; `_sbp_staggered_radial_pair`) satisfies the SBP
norm identity Pp·Dr = -(Pv·Gr)ᵀ + boundary-term with both operators
nullspace-clean, but in the full EVP it is necessary, not sufficient:
  * clean Dr, Gr used DIRECTLY  → checkerboard gone, damping ~1e-2
  * Div = -P⁻¹GᵀP (norm-adjoint) → damping ~1e-16, checkerboard back
For the inertial spectrum to be exactly imaginary the ENTIRE discrete
operator must be skew-symmetric under ONE consistent weighted inner
product ⟨x,y⟩ = x*Wy with W = blkdiag(Wv,Wv,Wv,Wp) — that includes the
Coriolis term, the metric factors (1/r, cscθ), AND the θ-derivative,
not just the radial Grad/Div. Mixing SBP-normed radial operators with
UNWEIGHTED angular/metric/Coriolis terms breaks global skewness
regardless of the radial choice. Even with an exact 1D radial closure
(G = -Hf⁻¹DᵀHc: machine-zero SBP residual, G sawtooth-content 0.026),
both pressure-coupling transposes carry a sawtooth once lifted to 2D
with the metric factors:
    G_rᵀ (radial) smallest-SV mode alternation = 1.40
    G_θᵀ (theta)  smallest-SV mode alternation = 1.83
The THETA coupling ((1/r)∂_θ p and (im/r sinθ)p on a COLLOCATED-θ grid)
has its OWN inf-sup checkerboard that radial-only staggering
structurally CANNOT remove, so the scalars need a 2D staggered grid.

A 2D C-grid (Arakawa-C: pressure at cell centers, v_r on r-faces, v_θ on
θ-faces, v_φ colocated with pressure) removes both defects only when it
is assembled adjoint-consistently. The Coriolis v_r↔v_φ and v_θ↔v_φ
couplings need INTERPOLATION between staggered locations; if the
forward and reverse averaging operators are built independently they are
not adjoint partners (mismatch 0.88 under the radial volume weight), the
Coriolis block is not skew, and both defects reappear (checkerboard
alternation 1.85, damping 4.5e-2). `build_primitive_inertial_evp_cgrid`
implements all three requirements: clean staggered gradients in both r
and θ; Div = -W_p⁻¹ Gᵀ W_v (the full norm-adjoint); and Coriolis
cross-component interpolations built as adjoint pairs (one averaging op
per coupling, its partner set to the weighted transpose). Verified on the
inertial-wave problem:
  * the analytic ℓ=m mode i/(m+1) is recovered to machine-precision
    Re(λ) ~ 1e-17 (UNDAMPED) and the correct frequency,
  * the mode CONVERGES with resolution (err 4e-3 → 2e-4, N: 24→48),
  * ~12/20 modes near a physical target are machine-undamped and the
    worst spurious |Re| (~5e-4) SHRINKS with N (degeneracy splitting,
    not a σ-tracking checkerboard floor).

IMPORTANT diagnostic caveat: the eigenvector node-to-node ALTERNATION
score (used to detect checkerboards on the COLOCATED grid) gives FALSE
POSITIVES on a staggered grid — v_r lives on r-faces, where a physical
mode legitimately has face-to-face structure that the colocated-tuned
threshold misreads as 'checkerboard'. The robust staggered-grid
diagnostics are: convergence to the analytic eigenvalue, machine-
precision Re(λ)=0 for undamped modes, and σ-independence (no clustering
at the shift target). The C-grid tests use these, not alternation.

# Pole treatment (parity, NOT Bekki's clamp)

Bekki 2022 (§2.4) uses the SAME 2D staggered layout as
build_primitive_inertial_evp_cgrid (Gilman 1975: v_r, v_θ on cell
edges; v_φ, ρ₁, s₁ cell-centered), with 2nd-order centered differences.
But their pole BC is a Dirichlet clamp (v_r=v_θ=v_φ=0, s₁=0 at the pole
for m≠0; a different θ-derivative set for m=0). spdee instead uses
PARITY conditions on each variable's staggered θ-sub-grid — derived from
Cartesian regularity, m-general (no m=0 vs m≠0 case split), and they
keep the pole regular without evaluating the singular 1/sinθ factors AT
the pole (scalars sit half a cell off it).

Parity facts (V19 spin basis v_± = (v_θ ∓ i v_φ)/√2, v_0 = v_r):
  * Spin component of weight s at azimuthal order m has pole-reflection
    parity (-1)^|m-s|.  (Confirmed three independent ways: Cartesian-
    smoothness numerics on two different smooth fields, and the analytic
    sin^|m-s| leading-power argument — all agree.)
        v_0 (=v_r): (-1)^|m|     v_+: (-1)^|m-1|     v_-: (-1)^|m+1|
    So v_+ and v_- share parity, OPPOSITE to v_0/scalars.
  * Pole-edge parity FOLD (ghost f(-θ)=parity·f(θ)) is accurate at the
    pole: even-parity → ~3rd order, odd-parity → 2nd order (a 2-pt
    staggered θ-diff), NO order loss at the singularity.
  * Staggered pole geometry is self-cleaning: put v_θ on the pole EDGE
    (θ=0) where the quadrature weight sinθ=0, so the pole node drops out
    of the weighted inner product and the adjoint divergence
    Div=-W⁻¹GᵀW handles it with NO special pole BC.
  * CONCRETE correctness win over Bekki: for m=1, v_+ stays FINITE at
    the pole (leading power sin^|m-1|=sin^0=const) while v_r→0 like sinθ.
    Bekki's "v_r=v_θ=v_φ=0 at pole" clamp wrongly forces v_+→0 for m=1 —
    over-constraining the physically important high-latitude m=1 mode.
    Parity conditions get this right.

Each transfer direction folds with its SOURCE field's parity:
center→edge ∂_θ (∇p → v_θ) uses scalar parity; edge→center ∂_θ (v_θ
divergence) uses v_θ parity. The parity-folded staggered θ pair IS
adjoint-consistent, and the apparent skewness-vs-accuracy tension at the
pole is illusory. Key facts (verified to machine precision):
  * With v_θ on θ-EDGES (pole edges at θ=0,π) and scalars on θ-CENTERS,
    and the natural volume weights W_e=sinθ_edge, W_c=sinθ_center, the
    norm-adjoint divergence D = -W_c⁻¹ Gᵀ W_e is IDENTICAL (rel. diff
    3.6e-16) to the physically-correct flux-form divergence
        (1/sinθ) ∂_θ(sinθ · v_θ).
  * The pole edge has W_e=sinθ=0, which zeroes the pole-edge v_θ
    coefficient in D — and this is PHYSICALLY CORRECT, because the flux
    sinθ·v_θ vanishes at the pole, so v_θ(pole) genuinely does not enter
    the divergence. No information is lost; no special pole BC needed.
  * So the SAME sinθ weight makes the operator (a) skew [no damping],
    (b) physically accurate [matches analytic to 1e-15 incl. pole cell],
    and (c) pole-regular [pole drops out automatically]. Geometry,
    physics, and skewness all align through the single sinθ norm — the
    deep reason the staggered grid with the natural volume norm is the
    right tool.

The singular m/sinθ Coriolis and pressure terms at the pole are benign
because of WHERE the layout places each variable:
  * v_φ and p both sit at θ-CENTERS (θ_half = (j+½)h), which NEVER touch
    the pole. So sinθ at centers has a finite floor (≈0.039 at the pole-
    adjacent center for N_θ=40) and 1/sinθ is BOUNDED (≈25), growing only
    like 1/(h/2) under refinement — large but finite and resolution-
    controlled, not singular.
  * The φ-momentum pressure coupling G_φ = (im/(r sinθ)) p is a PURE-
    IMAGINARY DIAGONAL (p, v_φ both at centers, no derivative). A pure-
    imaginary diagonal is AUTOMATICALLY skew-adjoint (-G* = G), so it
    lands on the imaginary axis with no special handling.
So the only variable that reaches the pole is v_θ (on edges), and there
the sinθ→0 weight is exactly the physically-correct zero-flux factor.
Each variable sits where its singular factors are benign — which is
precisely WHY Bekki's layout centers v_φ/p/ρ₁/s₁ and edges v_r/v_θ. The
layout is internally consistent.

The production path is therefore: 2D staggered C-grid (Bekki/Gilman
layout) + adjoint-consistent assembly (Div = norm-adjoint of clean G,
Coriolis interps as adjoint pairs) + parity-folded staggered θ with the
sinθ volume norm handling the poles automatically — no Dirichlet clamp,
m-general, and more correct than Bekki for the m=1 high-latitude mode.

# Physics

Inviscid incompressible inertial waves in a rotating sphere/shell,
in the co-rotating frame:

    λ v + 2 Ω₀ ẑ × v = -∇p,        ∇·v = 0

with Ω₀ = 1/2 (so the Coriolis coefficient 2Ω₀ = 1). The analytic
eigenvalues for the toroidal family are

    λ = i m / (ℓ(ℓ+1)),   ℓ = m, m+1, …

(Rieutord & Valdettaro 1997, Appendix B). For ℓ=m this is λ=i/(m+1).

# Why primitive variables


The companion test `test_evp_inertial_appendix_b.py` solves the SAME
physics in a toroidal-scalar formulation that ELIMINATES pressure —
precisely to dodge the LBB issue. Here we deliberately KEEP pressure
as a colocated unknown so the checkerboard appears, giving us a clean
problem on which to test whether radial staggering removes it.

# Discretization (colocated)

All four fields (v_r, v_θ, v_φ, p) live on the same (r, θ) nodes.
We use physical spherical components (not the spin basis) for
transparency. The radial derivative in ∇p and ∇·v uses a centered
standard stencil; on a colocated grid this stencil annihilates the
node-to-node sawtooth in p, producing the checkerboard null-space.

We work at fixed azimuthal order m; ∂_φ → i m. The equations in
spherical components (Ω₀ ẑ = Ω₀(cosθ r̂ − sinθ θ̂), 2Ω₀ = 1):

    λ v_r  + (ẑ×v)_r  = -∂_r p
    λ v_θ  + (ẑ×v)_θ  = -(1/r) ∂_θ p
    λ v_φ  + (ẑ×v)_φ  = -(im)/(r sinθ) p
    ∂_r v_r + (2/r) v_r + (1/r)∂_θ v_θ + (cotθ/r) v_θ + (im)/(r sinθ) v_φ = 0

with the Coriolis cross product (ẑ = cosθ r̂ − sinθ θ̂):
    ẑ × v = cosθ (r̂×v) − sinθ (θ̂×v)
          = cosθ (v_φ θ̂ − v_θ φ̂)·… 
Using r̂×θ̂=φ̂, r̂×φ̂=−θ̂, θ̂×φ̂=r̂:
    ẑ×v = cosθ(r̂×v) − sinθ(θ̂×v)
        r̂×v = v_θ φ̂ − v_φ θ̂
        θ̂×v = −v_r φ̂ + v_φ r̂
    ⇒ (ẑ×v)_r = −sinθ · v_φ
       (ẑ×v)_θ = −cosθ · v_φ
       (ẑ×v)_φ = cosθ · v_θ + sinθ · v_r

This module exposes builders so a sibling staggered-grid probe can
reuse the angular operators and only swap the radial ones.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.operators_core import lift_theta_op, diag_from_2d_array
from spdee.backends.fd2.stencils import standard_d1, _lagrange_derivative_weights


# ---------------------------------------------------------------------------
# Shared angular pieces (identical for colocated and staggered radial probes)
# ---------------------------------------------------------------------------

def _angular_ops(grid, m, order_theta):
    """Return the lifted angular first-derivative operator and the
    diagonal trig arrays needed by the inertial-wave system.

    All operators act on r-major-within-(r,θ) flattened fields of
    length N_r·N_θ.
    """
    N_r, N_theta = grid.N_r, grid.N_theta
    _, B_1_t = standard_d1(grid.theta, order=order_theta)
    B_1_t_lift = lift_theta_op(B_1_t, N_r)

    theta = grid.theta
    ones_r = np.ones(N_r)[:, None]
    sin_2d = (ones_r * np.sin(theta)[None, :]).astype(complex)
    cos_2d = (ones_r * np.cos(theta)[None, :]).astype(complex)
    cot_2d = (ones_r * (np.cos(theta) / np.sin(theta))[None, :]).astype(complex)
    csc_2d = (ones_r * (1.0 / np.sin(theta))[None, :]).astype(complex)
    r_2d = (grid.r[:, None] * np.ones(N_theta)[None, :]).astype(complex)
    inv_r_2d = (1.0 / grid.r)[:, None] * np.ones(N_theta)[None, :]
    inv_r_2d = inv_r_2d.astype(complex)

    return {
        "B_1_t_lift": B_1_t_lift,
        "sin": diag_from_2d_array(sin_2d),
        "cos": diag_from_2d_array(cos_2d),
        "cot": diag_from_2d_array(cot_2d),
        "csc": diag_from_2d_array(csc_2d),
        "r": diag_from_2d_array(r_2d),
        "inv_r": diag_from_2d_array(inv_r_2d),
        "N": N_r * N_theta,
    }


def _radial_d1_colocated(grid, order_r):
    """Colocated radial first-derivative operator (N×N), r-major."""
    N_theta = grid.N_theta
    _, B_1_r = standard_d1(grid.r, order=order_r)
    # Lift to (N_r·N_θ): the radial operator acts within each θ column.
    # r-major-within-(r,θ): flat index = i_r * N_theta + i_theta.
    # B_1_r is (N_r, N_r); we need it to act on the r index for each
    # fixed theta. That's a Kronecker: kron(B_1_r, I_theta).
    I_t = sp.eye(N_theta, format="csr", dtype=complex)
    return sp.kron(B_1_r, I_t, format="csr")


# ---------------------------------------------------------------------------
# Assemble the primitive inertial-wave EVP (colocated)
# ---------------------------------------------------------------------------

def build_primitive_inertial_evp(grid, m, order_r=4, order_theta=6,
                                 radial_d1=None, apply_bc=True):
    """Assemble (K, M) for the primitive inertial-wave EVP.

    State vector x = [v_r, v_θ, v_φ, p], each length N = N_r·N_θ,
    r-major. Equations (rows) in the same order: r-momentum,
    θ-momentum, φ-momentum, continuity.

    The EVP is  K x = λ M x  with M = diag(I, I, I, 0) (no time
    derivative on the continuity/pressure row → pressure is a
    constraint, the source of the LBB structure).

    Parameters
    ----------
    radial_d1 : sparse matrix or None
        The (N×N) radial first-derivative operator to use in ∇p and
        ∇·v. If None, uses the colocated standard stencil. A staggered
        probe passes its inter-grid operator(s) here.
    """
    A = _angular_ops(grid, m, order_theta)
    N = A["N"]
    Z = sp.csr_matrix((N, N), dtype=complex)
    I = sp.eye(N, format="csr", dtype=complex)
    im = 1j * m

    d_r = radial_d1 if radial_d1 is not None else _radial_d1_colocated(grid, order_r)
    B_t = A["B_1_t_lift"]
    sinD, cosD, cotD, cscD = A["sin"], A["cos"], A["cot"], A["csc"]
    inv_r = A["inv_r"]

    # --- Momentum rows ---
    # r:  λ v_r + (ẑ×v)_r = -∂_r p
    #     (ẑ×v)_r = -sinθ v_φ
    # ⇒  λ v_r = sinθ v_φ - ∂_r p
    # K row (LHS of K x = λ M x, i.e. the non-λ part moved to K):
    #   K: v_r row =  -(coriolis) - grad_p  acting to give λ v_r.
    # Write each momentum eq as:  λ v_i = RHS_i, so K row_i = RHS_i.
    # r-mom:  λ v_r = (sinθ) v_φ + (-∂_r) p
    K_vr = sp.hstack([Z, Z, sinD, -d_r], format="csr")
    # θ-mom:  λ v_θ = (cosθ) v_φ + (-(1/r)∂_θ) p
    #   (ẑ×v)_θ = -cosθ v_φ ⇒ λ v_θ = cosθ v_φ - (1/r)∂_θ p
    K_vt = sp.hstack([Z, Z, cosD, -(inv_r @ B_t)], format="csr")
    # φ-mom:  (ẑ×v)_φ = cosθ v_θ + sinθ v_r
    #   λ v_φ = -cosθ v_θ - sinθ v_r - (im)/(r sinθ) p
    K_vp = sp.hstack(
        [-sinD, -cosD, Z, -(im) * (inv_r @ cscD)], format="csr"
    )
    # --- Continuity row (constraint, no λ) ---
    #   ∂_r v_r + (2/r) v_r + (1/r)(∂_θ + cotθ) v_θ + (im)/(r sinθ) v_φ = 0
    cont = sp.hstack([
        d_r + 2 * inv_r,
        inv_r @ (B_t + cotD),
        im * (inv_r @ cscD),
        Z,
    ], format="csr")

    K = sp.vstack([K_vr, K_vt, K_vp, cont], format="csr")
    # M: identity on the three velocity rows, zero on continuity.
    M = sp.block_diag([I, I, I, Z], format="csr")
    if apply_bc:
        # v_r = 0 impenetrability at both radial walls (rows 0 and N_r-1).
        N_r, N_theta = grid.N_r, grid.N_theta
        K, M = _apply_vr_impenetrability(
            K, M, vr_offset=0, N_r=N_r, N_theta=N_theta,
            wall_rows=[0, N_r - 1],
        )
    return K.tocsr(), M.tocsr()


def solve_inertial_evp(K, M, sigma, k=20):
    """Manual shift-invert (same recipe as solve_evp)."""
    KmM = (K - sigma * M).tocsc()
    LU = spla.splu(KmM)
    LinOp = spla.LinearOperator(K.shape, matvec=lambda x: LU.solve(M @ x),
                                dtype=complex)
    try:
        mu = spla.eigs(LinOp, k=k, which="LM", return_eigenvectors=False,
                       maxiter=5000, tol=1e-10)
    except spla.ArpackNoConvergence as err:
        mu = err.eigenvalues
    finite = np.abs(mu) > 1e-12
    return 1.0 / mu[finite] + sigma


def _apply_vr_impenetrability(K, M, vr_offset, N_r, N_theta, wall_rows):
    """Replace the r-momentum rows at the radial walls with v_r = 0.

    Impenetrability `v_r = 0` is imposed by overwriting the r-momentum
    equation rows at the wall integer nodes with the identity on the
    v_r unknown, and zeroing the corresponding mass-matrix rows so the
    constraint carries no eigenvalue (these rows become λ-independent
    algebraic constraints, the standard BC-as-row-replacement used in
    spdee's solve_evp).

    Parameters
    ----------
    K, M : csr_matrix (modified in place via lil conversion, returned)
    vr_offset : int
        State-vector offset of the v_r block.
    N_r, N_theta : int
        Velocity-grid dimensions (integer nodes).
    wall_rows : iterable of int
        Radial indices to constrain (e.g. [0, N_r-1] for both walls).
    """
    Kl = K.tolil()
    Ml = M.tolil()
    for ir in wall_rows:
        for it in range(N_theta):
            row = vr_offset + ir * N_theta + it   # r-major within (r,θ)
            # Overwrite K row: v_r = 0  →  1·v_r(node) = 0.
            Kl.rows[row] = [row]
            Kl.data[row] = [1.0 + 0j]
            # Zero the mass row (algebraic constraint, no λ).
            Ml.rows[row] = []
            Ml.data[row] = []
    return Kl.tocsr(), Ml.tocsr()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def physical_eigenvalues(m, n_ell=4):
    """Analytic λ = i m/(ℓ(ℓ+1)) for ℓ = m … m+n_ell-1."""
    return np.array([1j * m / (ell * (ell + 1))
                     for ell in range(m, m + n_ell)], dtype=complex)


def count_checkerboard(lam, m, tol=0.02):
    """Split eigenvalues into (physical-matching, spurious).

    A computed λ is 'physical' if it lands within `tol` (relative) of
    one of the analytic i m/(ℓ(ℓ+1)) values; otherwise 'spurious'
    (the checkerboard cluster).
    """
    phys_targets = physical_eigenvalues(m, n_ell=8)
    physical, spurious = [], []
    for L in lam:
        if not np.isfinite(L):
            continue
        rel = np.min(np.abs(L - phys_targets)) / max(np.abs(L), 1e-12)
        (physical if rel < tol else spurious).append(L)
    return np.array(physical), np.array(spurious)


# ---------------------------------------------------------------------------
# Radial staggering: pressure on half-nodes, velocity on integer-nodes
# ---------------------------------------------------------------------------

def _intergrid_d1(src_nodes, dst_nodes, order=4):
    """(len(dst) × len(src)) matrix: f'(dst) ≈ D @ f(src).

    Inter-grid first derivative via local Lagrange weights. `src_nodes`
    and `dst_nodes` are the two staggered radial grids. Uses an
    `order+1`-node stencil centered on each destination point.
    """
    src = np.asarray(src_nodes, float)
    dst = np.asarray(dst_nodes, float)
    n_src = len(src)
    width = order + 1
    width = min(width, n_src)
    D = np.zeros((len(dst), n_src), dtype=complex)
    for i, xe in enumerate(dst):
        # Center the stencil on the source nodes nearest xe.
        j = np.searchsorted(src, xe)
        j0 = int(np.clip(j - width // 2, 0, n_src - width))
        idx = np.arange(j0, j0 + width)
        w = _lagrange_derivative_weights(src[idx], xe, order=1)
        D[i, idx] = w
    return sp.csr_matrix(D)


def _intergrid_interp(src_nodes, dst_nodes, order=4):
    """(len(dst) × len(src)) matrix: f(dst) ≈ P @ f(src) (interpolation).

    Needed for terms like (2/r) v_r evaluated on the half-node grid
    from v_r on integer nodes (and vice versa): the value, not the
    derivative, must be moved between grids.
    """
    src = np.asarray(src_nodes, float)
    dst = np.asarray(dst_nodes, float)
    n_src = len(src)
    width = min(order + 1, n_src)
    P = np.zeros((len(dst), n_src), dtype=complex)
    for i, xe in enumerate(dst):
        j = np.searchsorted(src, xe)
        j0 = int(np.clip(j - width // 2, 0, n_src - width))
        idx = np.arange(j0, j0 + width)
        # order=0 derivative weights = Lagrange interpolation weights.
        w = _lagrange_derivative_weights(src[idx], xe, order=0)
        P[i, idx] = w
    return sp.csr_matrix(P)


def _sbp_staggered_radial_pair(r_int, order=2):
    """Summation-by-parts staggered radial first-derivative PAIR.

    Returns (Dr, Gr, Pv, Pp):
      Dr : (N_rh × N_r)  divergence  (integer → half),  d/dr
      Gr : (N_r × N_rh)  gradient    (half → integer),  d/dr
      Pv : (N_r,)        diagonal norm (quadrature) on integer nodes
      Pp : (N_rh,)       diagonal norm (quadrature) on half nodes

    These are co-designed so the discrete SBP identity holds EXACTLY:

        Pp[:,None]*Dr  +  Gr.T*Pv[None,:]  =  boundary-only

    i.e. Pp·Dr = -(Pv·Gr)ᵀ except for the two physical end-point boundary
    terms. Both operators are individually nullspace-clean (simple ±1/h
    differences, no sawtooth annihilation). This is the 2nd-order
    diagonal-norm staggered SBP operator: forward difference for Dr,
    backward difference for Gr, with half-weight end norms on the
    integer (velocity) grid.

    Currently 2nd-order interior; the structure generalizes to higher
    order with wider co-designed boundary closures (future work).
    """
    if order != 2:
        raise NotImplementedError(
            "only the 2nd-order staggered SBP pair is implemented; "
            "higher order needs co-designed boundary closures.")
    r_int = np.asarray(r_int, float)
    N_r = len(r_int)
    N_rh = N_r - 1
    h = r_int[1] - r_int[0]
    if not np.allclose(np.diff(r_int), h):
        raise NotImplementedError(
            "2nd-order staggered SBP pair currently assumes a uniform "
            "radial grid.")
    # Divergence Dr: forward difference int→half, (D u)_j=(u_{j+1}-u_j)/h.
    Dr = np.zeros((N_rh, N_r), dtype=complex)
    for j in range(N_rh):
        Dr[j, j] = -1.0 / h
        Dr[j, j + 1] = 1.0 / h
    # Gradient Gr: backward difference half→int interior, (G p)_i=(p_i-p_{i-1})/h.
    Gr = np.zeros((N_r, N_rh), dtype=complex)
    for i in range(1, N_r - 1):
        Gr[i, i - 1] = -1.0 / h
        Gr[i, i] = 1.0 / h
    # Boundary gradient rows (i=0, i=N_r-1): one-sided. These rows are
    # overwritten by the v_r=0 impenetrability BC in the EVP, so their
    # exact form only needs to keep the operator well-defined.
    Gr[0, 0] = -1.0 / h
    Gr[0, 1] = 1.0 / h
    Gr[N_r - 1, N_rh - 2] = -1.0 / h
    Gr[N_r - 1, N_rh - 1] = 1.0 / h
    # Diagonal SBP norms.
    Pv = h * np.ones(N_r)
    Pv[0] = h / 2.0
    Pv[-1] = h / 2.0
    Pp = h * np.ones(N_rh)
    return Dr, Gr, Pv, Pp


def build_primitive_inertial_evp_sbp(grid, m, order_theta=6, apply_bc=True):
    """Inertial-wave EVP with a co-designed STAGGERED SBP Grad/Div pair.

    This is the resolution of the 2×2 study in the module docstring: it
    achieves all three required properties at once —
      (1) STAGGERED  (pressure on r-half-nodes) → no checkerboard,
      (2) discretely ADJOINT (Div = -P⁻¹GᵀP) → no spurious damping,
      (3) the radial pair is a co-designed SBP pair, so Gᵀ stays
          nullspace-clean → adjointness does NOT reintroduce the
          checkerboard (unlike transposing a plain-Lagrange gradient).

    Construction: build the COMPLETE gradient block G (all three
    velocity components: SBP-radial ∂_r, θ-derivative, and the algebraic
    φ term, each with their 1/r and cscθ metric factors), choose the
    single consistent 2D norm P = r²·sinθ · (SBP radial weights)·(θ
    weights), then set the ENTIRE divergence block to Div = -P_v⁻¹ Gᵀ P_p.
    No piecemeal mixing — the divergence is nothing but the norm-weighted
    adjoint of the gradient, so global discrete skewness holds exactly.
    """
    N_r, N_theta = grid.N_r, grid.N_theta
    r_int = grid.r
    r_half = 0.5 * (r_int[:-1] + r_int[1:])
    N_rh = N_r - 1
    Nv = N_r * N_theta
    Np = N_rh * N_theta
    theta = grid.theta
    im = 1j * m

    # SBP staggered radial pair.
    Dr, Gr, Pv_r, Pp_r = _sbp_staggered_radial_pair(r_int, order=2)
    I_t = sp.eye(N_theta, format="csr", dtype=complex)
    Gr_f = sp.kron(sp.csr_matrix(Gr), I_t, format="csr")   # half→int radial grad

    # θ derivative (collocated; its transpose is the standard d1 transpose).
    _, B_1_t = standard_d1(grid.theta, order=order_theta)
    I_half_r = sp.eye(N_rh, format="csr", dtype=complex)
    B_t_half = sp.kron(I_half_r, B_1_t, format="csr")      # ∂_θ on half-node p
    # Interpolate p (half-r) to integer-r so θ-grad and φ term live on
    # the velocity (integer) radial nodes.
    interp_half_to_int = sp.kron(
        _intergrid_interp(r_half, r_int, 4), I_t, format="csr")

    def diag_trig(rvals, fn):
        arr = (np.ones(len(rvals))[:, None] * fn(theta)[None, :]).astype(complex)
        return diag_from_2d_array(arr)

    sin_i = diag_trig(r_int, np.sin)
    cos_i = diag_trig(r_int, np.cos)
    csc_i = diag_trig(r_int, lambda t: 1.0 / np.sin(t))
    inv_r_i = diag_from_2d_array(
        ((1.0 / r_int)[:, None] * np.ones(N_theta)[None, :]).astype(complex))

    # --- FULL gradient block G: (3 Nv) × Np ---
    G_r = Gr_f                                          # ∂_r p
    G_t = inv_r_i @ (interp_half_to_int @ B_t_half)     # (1/r) ∂_θ p
    G_p = (im) * (inv_r_i @ csc_i) @ interp_half_to_int  # (im)/(r sinθ) p
    G = sp.vstack([G_r, G_t, G_p], format="csr")

    # --- Single consistent 2D norm P = r² sinθ · w_r · w_θ ---
    def theta_weights(tv):
        w = np.zeros(len(tv))
        w[1:-1] = 0.5 * (tv[2:] - tv[:-2])
        w[0] = 0.5 * (tv[1] - tv[0])
        w[-1] = 0.5 * (tv[-1] - tv[-2])
        return w
    wt = theta_weights(theta)
    # velocity-grid (integer) node weights and pressure-grid (half) weights,
    # using the SBP radial norms Pv_r, Pp_r.
    def vol_diag(rv, wr):
        wv2d = (rv**2 * wr)[:, None] * (np.sin(theta) * wt)[None, :]
        return wv2d.ravel().astype(complex)
    Wp_diag = vol_diag(r_half, Pp_r)        # Np
    Wv_one = vol_diag(r_int, Pv_r)          # Nv (per component)
    Wv = sp.block_diag(
        [sp.diags(Wv_one), sp.diags(Wv_one), sp.diags(Wv_one)], format="csr")
    Wp_inv = sp.diags(1.0 / Wp_diag, format="csr")

    # --- Divergence = negative norm-weighted adjoint of G ---
    Div = -(Wp_inv @ G.conj().T @ Wv)       # Np × (3 Nv)

    # --- Assemble ---
    Z_vv = sp.csr_matrix((Nv, Nv), dtype=complex)
    I_v = sp.eye(Nv, format="csr", dtype=complex)
    Z_pp = sp.csr_matrix((Np, Np), dtype=complex)

    K_vr = sp.hstack([Z_vv, Z_vv, sin_i, -G_r], format="csr")
    K_vt = sp.hstack([Z_vv, Z_vv, cos_i, -G_t], format="csr")
    K_vp = sp.hstack([-sin_i, -cos_i, Z_vv, -G_p], format="csr")
    cont = sp.hstack([Div, Z_pp], format="csr")

    K = sp.vstack([K_vr, K_vt, K_vp, cont], format="csr")
    M = sp.block_diag([I_v, I_v, I_v, Z_pp], format="csr")
    if apply_bc:
        K, M = _apply_vr_impenetrability(
            K, M, vr_offset=0, N_r=N_r, N_theta=N_theta,
            wall_rows=[0, N_r - 1])
    return K.tocsr(), M.tocsr()


def build_primitive_inertial_evp_staggered_adjoint(grid, m, order_r=4,
                                                   order_theta=6, apply_bc=True):
    """Adjoint-consistent staggered inertial-wave EVP.

    Same radial staggering as `build_primitive_inertial_evp_staggered`,
    but the continuity (divergence) block is built as the discrete
    negative-adjoint of the full gradient block under the physical L²
    inner product with volume weight dV = r² sinθ dr dθ:

        ⟨v, ∇p⟩_V = -⟨∇·v, p⟩_V   (no-penetration boundary)
        ⇒  Div = -W_p^{-1} Gᵀ W_v

    where G is the FULL gradient operator (pressure-state → velocity-
    state, all three spin/physical components with their metric
    factors), W_v and W_p are the diagonal quadrature-weight matrices on
    the velocity (integer-node) and pressure (half-node) grids. This
    guarantees the pressure coupling is exactly skew, so the inertial
    spectrum is purely imaginary up to roundoff — no spurious ±Re
    splitting from a mismatched Grad/Div pair.

    The gradient block G has three row-groups (the three momentum
    components' pressure-gradient terms):
        G_r p = ∂_r p                       (half→int radial d1)
        G_θ p = (1/r) ∂_θ p                 (θ deriv, then interp half→int radius)
        G_φ p = (im)/(r sinθ) p             (algebraic, interp half→int radius)
    """
    N_r, N_theta = grid.N_r, grid.N_theta
    r_int = grid.r
    r_half = 0.5 * (r_int[:-1] + r_int[1:])
    N_rh = N_r - 1
    Nv = N_r * N_theta
    Np = N_rh * N_theta
    theta = grid.theta
    im = 1j * m

    _, B_1_t = standard_d1(grid.theta, order=order_theta)
    I_t = sp.eye(N_theta, format="csr", dtype=complex)
    I_half_r = sp.eye(N_rh, format="csr", dtype=complex)
    B_t_half = sp.kron(I_half_r, B_1_t, format="csr")  # ∂_θ on half-node fields

    def diag_trig(rvals, fn):
        arr = (np.ones(len(rvals))[:, None] * fn(theta)[None, :]).astype(complex)
        return diag_from_2d_array(arr)

    sin_i = diag_trig(r_int, np.sin)
    cos_i = diag_trig(r_int, np.cos)
    csc_i = diag_trig(r_int, lambda t: 1.0 / np.sin(t))
    inv_r_i = diag_from_2d_array(
        ((1.0 / r_int)[:, None] * np.ones(N_theta)[None, :]).astype(complex))

    # --- Build the FULL gradient block G: (3 Nv) × Np ---
    # Maps p (on half nodes) to the three velocity-component gradient
    # contributions (on integer nodes).
    d_r_half_to_int = sp.kron(_intergrid_d1(r_half, r_int, order_r), I_t, format="csr")
    interp_half_to_int = sp.kron(
        _intergrid_interp(r_half, r_int, order_r), I_t, format="csr")
    # G_r = ∂_r p
    G_r = d_r_half_to_int
    # G_θ = (1/r) ∂_θ p, with ∂_θ p computed on half nodes then interp to int radius.
    G_t = inv_r_i @ (interp_half_to_int @ B_t_half)
    # G_φ = (im)/(r sinθ) p, interp p to integer radius.
    G_p = (im) * (inv_r_i @ csc_i) @ interp_half_to_int
    G = sp.vstack([G_r, G_t, G_p], format="csr")   # (3Nv) × Np

    # --- Volume-weight matrices (diagonal) ---
    # dV = r² sinθ dr dθ. On a tensor grid the per-node quadrature
    # weight ∝ r² sinθ times the local grid spacings. The dr/dθ spacings
    # are common to the inner product on both sides for the SAME θ grid;
    # what differs between the velocity and pressure grids is the radial
    # node set (integer vs half) and we must use trapezoidal radial
    # weights consistent with each. For the adjoint identity we need
    # consistent quadrature, so use composite weights:
    def radial_weights(rv):
        # trapezoidal weights for ∫ f(r) dr on node set rv
        w = np.zeros(len(rv))
        w[1:-1] = 0.5 * (rv[2:] - rv[:-2])
        w[0] = 0.5 * (rv[1] - rv[0])
        w[-1] = 0.5 * (rv[-1] - rv[-2])
        return w
    def theta_weights(tv):
        w = np.zeros(len(tv))
        w[1:-1] = 0.5 * (tv[2:] - tv[:-2])
        w[0] = 0.5 * (tv[1] - tv[0])
        w[-1] = 0.5 * (tv[-1] - tv[-2])
        return w

    wr_int = radial_weights(r_int)
    wr_half = radial_weights(r_half)
    wt = theta_weights(theta)
    # 2D node weight ∝ r² sinθ · w_r · w_θ.
    def vol_diag(rv, wr):
        wv2d = (rv**2 * wr)[:, None] * (np.sin(theta) * wt)[None, :]
        return sp.diags(wv2d.ravel().astype(complex), format="csr")
    Wp = vol_diag(r_half, wr_half)        # Np × Np
    Wv_one = vol_diag(r_int, wr_int)      # Nv × Nv  (per velocity component)
    # Velocity inner product is the sum over 3 components, each with the
    # same scalar volume weight, so W_v = blkdiag(Wv_one, Wv_one, Wv_one).
    Wv = sp.block_diag([Wv_one, Wv_one, Wv_one], format="csr")

    # --- Divergence as negative adjoint of G ---
    Wp_inv = sp.diags(1.0 / Wp.diagonal(), format="csr")
    Div = -(Wp_inv @ G.conj().T @ Wv)     # Np × (3Nv)

    # --- Assemble momentum rows (Coriolis + gradient) ---
    Z_vv = sp.csr_matrix((Nv, Nv), dtype=complex)
    I_v = sp.eye(Nv, format="csr", dtype=complex)
    Z_pp = sp.csr_matrix((Np, Np), dtype=complex)

    # r:  λ v_r = sinθ v_φ - G_r p
    K_vr = sp.hstack([Z_vv, Z_vv, sin_i, -G_r], format="csr")
    # θ:  λ v_θ = cosθ v_φ - G_θ p
    K_vt = sp.hstack([Z_vv, Z_vv, cos_i, -G_t], format="csr")
    # φ:  λ v_φ = -sinθ v_r - cosθ v_θ - G_φ p
    K_vp = sp.hstack([-sin_i, -cos_i, Z_vv, -G_p], format="csr")
    # continuity: Div · v = 0  (Div already includes the 3 vel blocks)
    cont = sp.hstack([Div, Z_pp], format="csr")

    K = sp.vstack([K_vr, K_vt, K_vp, cont], format="csr")
    M = sp.block_diag([I_v, I_v, I_v, Z_pp], format="csr")
    if apply_bc:
        K, M = _apply_vr_impenetrability(
            K, M, vr_offset=0, N_r=N_r, N_theta=N_theta,
            wall_rows=[0, N_r - 1],
        )
    return K.tocsr(), M.tocsr()


# ---------------------------------------------------------------------------
# Full 2D staggered grid (Arakawa-C), assembled adjoint-consistently
# ---------------------------------------------------------------------------

def _stag_diff_1d(n_face, n_cell, h):
    """2-point staggered first difference, cell→face, shape (n_face, n_cell).

    Interior face i sits between cells i-1 and i: (f_i - f_{i-1})/h.
    Endpoint faces use the single adjacent cell (overwritten by BC where
    relevant); kept simple so the operator stays nullspace-clean.
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


def _avg_1d(n_to, n_from):
    """2-point averaging interpolation, shape (n_to, n_from).

    Used for Coriolis cross-component transfers. The REVERSE-direction
    partner must be built as the weighted transpose of this (not an
    independent average) so the Coriolis block stays skew.
    """
    P = np.zeros((n_to, n_from), dtype=complex)
    for i in range(1, n_to - 1):
        P[i, i - 1] = 0.5
        P[i, i] = 0.5
    P[0, 0] = 1.0
    P[n_to - 1, n_from - 1] = 1.0
    return sp.csr_matrix(P)


def build_primitive_inertial_evp_cgrid(grid, m, apply_bc=True):
    """Inertial-wave EVP on a full 2D staggered (Arakawa-C) grid,
    assembled to be SIMULTANEOUSLY checkerboard-free and skew.

    Implements the specification distilled from the 2×2 study + the
    radial-only / SBP / first-C-grid attempts (see module docstring):

      LAYOUT (Arakawa-C):
        p   at cell centers   (r_half, θ_half)
        v_r at r-faces        (r_int,  θ_half)
        v_θ at θ-faces        (r_half, θ_int)
        v_φ at cell centers   (r_half, θ_half)   [colocated with p]

      REQUIREMENT 1 — clean staggered gradients in BOTH directions:
        G_r = ∂_r p   (cell→r-face, 2-pt staggered r-diff)
        G_θ = (1/r)∂_θ p (cell→θ-face, 2-pt staggered θ-diff)
        G_φ = (im/(r sinθ)) p  (local: v_φ colocated with p)
      Each is a short staggered difference (or local), hence
      nullspace-clean.

      REQUIREMENT 2 — divergence is the norm-adjoint of the clean G:
        Div = -W_v^{-1} Gᵀ W_p
      with W the r²sinθ volume weights on each sub-grid. Because G is
      clean in both directions, Gᵀ is a clean divergence in both — so
      adjointness (⇒ skew ⇒ no spurious damping) does NOT reintroduce a
      checkerboard. This is the combination radial-only could not reach,
      because radial-only left θ collocated and its θ-coupling transpose
      carried a sawtooth.

      REQUIREMENT 3 — Coriolis cross-component interpolations as ADJOINT
      PAIRS. The Coriolis term couples v_r, v_θ, v_φ which live at three
      different staggered locations, so interpolation is unavoidable.
      For Coriolis to stay skew (it is the rotation generator ẑ×, an
      antisymmetric operator) each forward interpolation P_{a→b} is built
      once and its reverse partner is set to the weighted transpose
      P_{b→a} = W_b^{-1} P_{a→b}ᵀ W_a. Building the reverse as an
      independent average breaks
      skewness and reintroduces O(1e-2) damping.
    """
    N_r, N_theta = grid.N_r, grid.N_theta
    r_int = grid.r
    r_half = 0.5 * (r_int[:-1] + r_int[1:])
    # θ on this probe is cell-centered (poles excluded). Treat the given
    # grid.theta as the θ_int (face) locations and form θ_half between them.
    th_int = grid.theta
    th_half = 0.5 * (th_int[:-1] + th_int[1:])
    Nr_i, Nr_h = N_r, N_r - 1
    Nt_i, Nt_h = N_theta, N_theta - 1
    hr = r_int[1] - r_int[0]
    ht = th_int[1] - th_int[0]
    im = 1j * m

    # Sub-grid sizes.
    n_p = Nr_h * Nt_h          # p   at (r_half, θ_half)
    n_vr = Nr_i * Nt_h         # v_r at (r_int,  θ_half)
    n_vt = Nr_h * Nt_i         # v_θ at (r_half, θ_int)
    n_vp = Nr_h * Nt_h         # v_φ at (r_half, θ_half)

    # 1D staggered diffs / identities.
    Gr_1d = _stag_diff_1d(Nr_i, Nr_h, hr)     # cell→r-face in r
    Gt_1d = _stag_diff_1d(Nt_i, Nt_h, ht)     # cell→θ-face in θ
    I_th_half = sp.eye(Nt_h, format="csr", dtype=complex)
    I_rh = sp.eye(Nr_h, format="csr", dtype=complex)

    def diagrt(rv, tv, fn):
        a = fn(rv[:, None], tv[None, :])
        return sp.diags(np.asarray(a).ravel().astype(complex), format="csr")

    # --- REQUIREMENT 1: clean gradient components ---
    # G_r : p(r_half,θ_half) → v_r(r_int,θ_half).  ∂_r, staggered in r.
    G_r = sp.kron(Gr_1d, I_th_half, format="csr")
    # G_θ : p(r_half,θ_half) → v_θ(r_half,θ_int).  (1/r)∂_θ, staggered in θ.
    inv_r_vt = diagrt(r_half, th_int, lambda r, t: 1.0 / r + 0 * t)
    G_t = inv_r_vt @ sp.kron(I_rh, Gt_1d, format="csr")
    # G_φ : p(r_half,θ_half) → v_φ(r_half,θ_half).  (im/(r sinθ)), local.
    G_p = diagrt(r_half, th_half, lambda r, t: im / (r * np.sin(t)))
    G = sp.vstack([G_r, G_t, G_p], format="csr")     # (n_vr+n_vt+n_vp) × n_p

    # --- volume-weight norms on each sub-grid: r² sinθ · h_r · h_θ ---
    def vol(rv, tv):
        a = (rv**2)[:, None] * np.sin(tv)[None, :]
        return a.ravel().astype(complex) * hr * ht
    W_vr = vol(r_int, th_half)
    W_vt = vol(r_half, th_int)
    W_vp = vol(r_half, th_half)
    W_p = vol(r_half, th_half)
    W_v = np.concatenate([W_vr, W_vt, W_vp])

    # --- REQUIREMENT 2: divergence = -W_p^{-1} Gᵀ W_v ---
    # G: pressure(n_p) → velocity(n_v), so Gᵀ: velocity → pressure, and the
    # norm-adjoint divergence is Div = -W_p^{-1} Gᵀ W_v  (n_p × n_v).
    Div = -(sp.diags(1.0 / W_p) @ G.conj().T @ sp.diags(W_v))   # n_p × n_v

    # --- REQUIREMENT 3: Coriolis as adjoint pairs ---
    # (ẑ×v)_r = -sinθ v_φ          [on v_r grid; v_φ at cell ctr → r-face]
    # (ẑ×v)_θ = -cosθ v_φ          [on v_θ grid; v_φ at cell ctr → θ-face]
    # (ẑ×v)_φ =  cosθ v_θ + sinθ v_r  [on v_φ grid; v_θ,v_r → cell ctr]
    #
    # Forward interpolations (built once):
    #   P_vp2vr : v_φ(r_half,θ_half) → v_r(r_int,θ_half)   [avg in r]
    #   P_vp2vt : v_φ(r_half,θ_half) → v_θ(r_half,θ_int)   [avg in θ]
    # Reverse partners set to weighted transpose so Coriolis is skew.
    P_vp2vr = sp.kron(_avg_1d(Nr_i, Nr_h), I_th_half, format="csr")
    P_vp2vt = sp.kron(I_rh, _avg_1d(Nt_i, Nt_h), format="csr")
    # Weighted-transpose reverse partners:
    #   P_vr2vp = W_vp^{-1} P_vp2vrᵀ W_vr   (v_r → v_φ)
    #   P_vt2vp = W_vp^{-1} P_vp2vtᵀ W_vt   (v_θ → v_φ)
    P_vr2vp = sp.diags(1.0 / W_vp) @ P_vp2vr.conj().T @ sp.diags(W_vr)
    P_vt2vp = sp.diags(1.0 / W_vp) @ P_vp2vt.conj().T @ sp.diags(W_vt)

    sin_vr = diagrt(r_int, th_half, lambda r, t: np.sin(t) + 0 * r)
    cos_vt = diagrt(r_half, th_int, lambda r, t: np.cos(t) + 0 * r)
    sin_vp = diagrt(r_half, th_half, lambda r, t: np.sin(t) + 0 * r)
    cos_vp = diagrt(r_half, th_half, lambda r, t: np.cos(t) + 0 * r)

    Cor_r_vp = -sin_vr @ P_vp2vr          # v_r eq ← v_φ
    Cor_t_vp = -cos_vt @ P_vp2vt          # v_θ eq ← v_φ
    Cor_p_vr = sin_vp @ P_vr2vp           # v_φ eq ← v_r
    Cor_p_vt = cos_vp @ P_vt2vp           # v_φ eq ← v_θ

    # --- Assemble K, M ---
    Z_rr = sp.csr_matrix((n_vr, n_vr), dtype=complex)
    Z_rt = sp.csr_matrix((n_vr, n_vt), dtype=complex)
    Z_tt = sp.csr_matrix((n_vt, n_vt), dtype=complex)
    Z_tr = sp.csr_matrix((n_vt, n_vr), dtype=complex)
    Z_pp = sp.csr_matrix((n_vp, n_vp), dtype=complex)
    Z_cont = sp.csr_matrix((n_p, n_p), dtype=complex)

    # rows: v_r, v_θ, v_φ, continuity(p); cols: v_r, v_θ, v_φ, p
    G_r_neg, G_t_neg, G_p_neg = -G_r, -G_t, -G_p
    K_vr = sp.hstack([Z_rr, Z_rt, Cor_r_vp, G_r_neg], format="csr")
    K_vt = sp.hstack([Z_tr, Z_tt, Cor_t_vp, G_t_neg], format="csr")
    K_vp = sp.hstack([Cor_p_vr, Cor_p_vt, Z_pp, G_p_neg], format="csr")
    cont = sp.hstack([Div, Z_cont], format="csr")
    K = sp.vstack([K_vr, K_vt, K_vp, cont], format="csr")

    n_v = n_vr + n_vt + n_vp
    M = sp.block_diag([sp.eye(n_v, format="csr", dtype=complex), Z_cont],
                      format="csr")

    if apply_bc:
        # v_r = 0 at the radial walls. v_r lives on r-faces (r_int), so
        # the wall rows are the first and last radial face index, all θ_half.
        Kl = K.tolil()
        Ml = M.tolil()
        for ir in (0, Nr_i - 1):
            for it in range(Nt_h):
                row = ir * Nt_h + it     # v_r block starts at offset 0
                Kl.rows[row] = [row]
                Kl.data[row] = [1.0 + 0j]
                Ml.rows[row] = []
                Ml.data[row] = []
        K, M = Kl.tocsr(), Ml.tocsr()

    return K.tocsr(), M.tocsr(), (n_vr, n_vt, n_vp, n_p), (Nr_i, Nt_h)


def build_primitive_inertial_evp_staggered(grid, m, order_r=4, order_theta=6,
                                           apply_bc=True):
    """Assemble (K, M) for the primitive inertial-wave EVP with RADIAL
    staggering: pressure on r-half-nodes, velocity on r-integer-nodes.

    State x = [v_r, v_θ, v_φ (each on integer nodes, length N_v=N_r·N_θ),
               p (on half nodes, length N_p=(N_r-1)·N_θ)].
    Momentum rows live on integer nodes; continuity on half nodes.

    The radial pressure gradient ∂_r p maps half→integer (an inter-grid
    d1, N_r×(N_r-1) radially). The radial velocity divergence ∂_r v_r
    maps integer→half ((N_r-1)×N_r). θ stays colocated; trig
    coefficients are evaluated at the appropriate radii for each row set.
    """
    N_r, N_theta = grid.N_r, grid.N_theta
    r_int = grid.r
    r_half = 0.5 * (r_int[:-1] + r_int[1:])
    N_rh = N_r - 1
    Nv = N_r * N_theta            # velocity block size
    Np = N_rh * N_theta           # pressure block size

    # Angular operator (shared θ), lifted to each radial grid.
    _, B_1_t = standard_d1(grid.theta, order=order_theta)
    I_int_r = sp.eye(N_r, format="csr", dtype=complex)
    I_half_r = sp.eye(N_rh, format="csr", dtype=complex)
    B_t_int = sp.kron(I_int_r, B_1_t, format="csr")     # acts on integer-node fields
    B_t_half = sp.kron(I_half_r, B_1_t, format="csr")   # acts on half-node fields

    theta = grid.theta
    def diag_trig(rvals, fn):
        arr = (np.ones(len(rvals))[:, None] * fn(theta)[None, :]).astype(complex)
        return diag_from_2d_array(arr)
    # On integer nodes (momentum rows):
    sin_i = diag_trig(r_int, np.sin)
    cos_i = diag_trig(r_int, np.cos)
    csc_i = diag_trig(r_int, lambda t: 1.0 / np.sin(t))
    inv_r_i = diag_from_2d_array(
        ((1.0 / r_int)[:, None] * np.ones(N_theta)[None, :]).astype(complex))
    # On half nodes (continuity row):
    cot_h = diag_trig(r_half, lambda t: np.cos(t) / np.sin(t))
    csc_h = diag_trig(r_half, lambda t: 1.0 / np.sin(t))
    inv_r_h = diag_from_2d_array(
        ((1.0 / r_half)[:, None] * np.ones(N_theta)[None, :]).astype(complex))

    im = 1j * m
    # Radial inter-grid operators (lifted over θ via Kron with I_theta).
    I_t = sp.eye(N_theta, format="csr", dtype=complex)
    d_r_half_to_int = sp.kron(_intergrid_d1(r_half, r_int, order_r), I_t, format="csr")   # ∂_r p
    d_r_int_to_half = sp.kron(_intergrid_d1(r_int, r_half, order_r), I_t, format="csr")   # ∂_r v_r
    interp_int_to_half = sp.kron(_intergrid_interp(r_int, r_half, order_r), I_t, format="csr")

    Z_vv = sp.csr_matrix((Nv, Nv), dtype=complex)
    Z_pp = sp.csr_matrix((Np, Np), dtype=complex)
    I_v = sp.eye(Nv, format="csr", dtype=complex)

    # --- Momentum rows (on integer nodes) ---
    # r:  λ v_r = sinθ v_φ - ∂_r p
    K_vr = sp.hstack([Z_vv, Z_vv, sin_i, -d_r_half_to_int], format="csr")
    # θ:  λ v_θ = cosθ v_φ - (1/r)∂_θ p
    #     ∂_θ p must be interpolated to integer nodes: B_t on half-node p,
    #     then interp half→int. Build (1/r_i) · interp_int? No: p lives on
    #     half nodes, B_t_half gives ∂_θ p on half nodes, interpolate to int.
    interp_half_to_int = sp.kron(
        _intergrid_interp(r_half, r_int, order_r), I_t, format="csr")
    dtheta_p_at_int = interp_half_to_int @ B_t_half
    K_vt = sp.hstack([Z_vv, Z_vv, cos_i, -(inv_r_i @ dtheta_p_at_int)], format="csr")
    # φ:  λ v_φ = -sinθ v_r - cosθ v_θ - (im)/(r sinθ) p
    p_at_int = interp_half_to_int
    K_vp = sp.hstack(
        [-sin_i, -cos_i, Z_vv, -(im) * (inv_r_i @ csc_i) @ p_at_int],
        format="csr")
    # --- Continuity row (on half nodes) ---
    #   ∂_r v_r + (2/r) v_r + (1/r)(∂_θ + cotθ) v_θ + (im)/(r sinθ) v_φ = 0
    #   v_r, v_θ, v_φ on integer nodes → interpolate value terms to half nodes.
    cont = sp.hstack([
        d_r_int_to_half + 2 * (inv_r_h @ interp_int_to_half),
        inv_r_h @ (interp_int_to_half @ B_t_int + cot_h @ interp_int_to_half),
        im * (inv_r_h @ csc_h) @ interp_int_to_half,
        Z_pp,
    ], format="csr")

    K = sp.bmat([
        [K_vr[:, :Nv], K_vr[:, Nv:2*Nv], K_vr[:, 2*Nv:3*Nv], K_vr[:, 3*Nv:]],
    ], format="csr") if False else sp.vstack([K_vr, K_vt, K_vp, cont], format="csr")

    M = sp.block_diag([I_v, I_v, I_v, Z_pp], format="csr")
    if apply_bc:
        # v_r = 0 at both walls. This overwrites the r-momentum rows at
        # the integer endpoints — which are precisely the rows whose
        # ∂_r p needed extrapolation off the half-node grid. So the BC
        # both supplies the physics and removes the polluting
        # extrapolation rows. Pressure (on interior half-nodes) needs
        # no wall BC.
        K, M = _apply_vr_impenetrability(
            K, M, vr_offset=0, N_r=N_r, N_theta=N_theta,
            wall_rows=[0, N_r - 1],
        )
    return K.tocsr(), M.tocsr()


def _spin_parity(m, s):
    """Pole-reflection parity of the spin-weight-s component at order m.

    V19 basis (v_± = (v_θ ∓ i v_φ)/√2, v_0 = v_r): spin weights are
    v_0 → s=0, v_+ → s=+1, v_- → s=-1. The parity under pole reflection
    is (-1)^|m-s|, derived from Cartesian regularity (a spin-weight-s,
    order-m field behaves like sin^|m-s|θ near the pole). The production
    parity-aware staggered θ-stencil folds ghosts with these signs.
    """
    return (-1) ** abs(m - s)


class TestStaggeredThetaPoleAdjoint:
    """The parity-folded staggered θ pair is adjoint-consistent at the
    pole, and the norm-adjoint divergence equals the physical flux-form
    divergence (1/sinθ)∂_θ(sinθ v_θ) — so the sinθ norm makes the
    operator simultaneously skew, accurate, and pole-regular."""

    def _grids(self, Nc):
        h = np.pi / Nc
        c = (np.arange(Nc) + 0.5) * h     # θ-centers (scalars / v_φ)
        e = np.arange(Nc + 1) * h         # θ-edges (v_θ), poles at e[0]=0,e[-1]=π
        return h, c, e

    def test_norm_adjoint_equals_physical_flux_divergence(self):
        Nc = 40
        h, c, e = self._grids(Nc)
        sin_e, sin_c = np.sin(e), np.sin(c)
        # Gradient centers→edges (∂_θ); pole edges carry zero flux weight.
        G = np.zeros((Nc + 1, Nc))
        for i in range(1, Nc):
            G[i, i - 1] = -1 / h
            G[i, i] = 1 / h
        G[0, 0] = 0.0
        G[Nc, Nc - 1] = 0.0
        # Physical flux-form divergence (1/sinθ)∂_θ(sinθ v_θ).
        D_phys = np.zeros((Nc, Nc + 1))
        for j in range(Nc):
            D_phys[j, j] = -sin_e[j] / (sin_c[j] * h)
            D_phys[j, j + 1] = sin_e[j + 1] / (sin_c[j] * h)
        # Norm-adjoint divergence with sinθ volume weights.
        D_adj = -np.diag(1.0 / sin_c) @ G.T @ np.diag(sin_e)
        rel = np.linalg.norm(D_adj - D_phys) / np.linalg.norm(D_phys)
        assert rel < 1e-12, f"adjoint != physical flux divergence (rel {rel:.1e})"

    def test_pole_edge_carries_zero_weight(self):
        """The pole-edge v_θ coefficient vanishes (flux sinθ·v_θ → 0),
        so the pole drops out without any special boundary condition."""
        Nc = 40
        h, c, e = self._grids(Nc)
        assert np.isclose(np.sin(e[0]), 0.0) and np.isclose(np.sin(e[-1]), 0.0)
        # Physical divergence pole-cell row uses only the interior edge.
        sin_e, sin_c = np.sin(e), np.sin(c)
        pole_coef = sin_e[0] / (sin_c[0] * h)   # coefficient on the pole edge
        assert np.isclose(pole_coef, 0.0), "pole edge should carry zero weight"


class TestSpinBasisPoleParity:
    """Lock in the verified spin-component pole parities (V19 basis).

    Cross-checked three ways in development (Cartesian-smoothness numerics
    on two fields + the analytic sin^|m-s| argument). These signs drive
    the production parity-aware staggered θ-stencil that replaces Bekki's
    Dirichlet pole clamp.
    """

    def test_parity_formula_matches_cartesian_smoothness(self):
        """The (-1)^|m-s| formula matches the parity of a smooth Cartesian
        vector field's m-th azimuthal spin components across the pole."""
        def field(x, y, z):
            return (x * z + 0.3 * y**2 + 0.5,
                    y * z - 0.2 * x**2 + 0.1,
                    1.0 + 0.4 * x * y + z**2)

        def spin_at(theta, m):
            Nph = 256
            ph = np.linspace(0, 2 * np.pi, Nph, endpoint=False)
            x = np.sin(theta) * np.cos(ph)
            y = np.sin(theta) * np.sin(ph)
            z = np.cos(theta) * np.ones_like(ph)
            Vx, Vy, Vz = field(x, y, z)
            vr = (np.sin(theta) * np.cos(ph) * Vx
                  + np.sin(theta) * np.sin(ph) * Vy + np.cos(theta) * Vz)
            vt = (np.cos(theta) * np.cos(ph) * Vx
                  + np.cos(theta) * np.sin(ph) * Vy - np.sin(theta) * Vz)
            vp = -np.sin(ph) * Vx + np.cos(ph) * Vy
            fm = lambda f: np.sum(f * np.exp(-1j * m * ph)) / Nph
            v0 = fm(vr)
            vplus = (fm(vt) - 1j * fm(vp)) / np.sqrt(2)
            vminus = (fm(vt) + 1j * fm(vp)) / np.sqrt(2)
            return v0, vplus, vminus

        for m in (0, 1, 2, 3, 8):
            th = 0.2
            a = spin_at(th, m)
            b = spin_at(-th, m)   # pole-reflected
            # measured parity = sign of ratio (reflected / original)
            for comp, s in zip(a, (0, 1, -1)):
                # match against the corresponding reflected component
                pass
            v0a, vpa, vma = a
            v0b, vpb, vmb = b
            # Skip components with no Fourier content at this m (a low-order
            # polynomial field has no high-m signal; the ratio is then pure
            # roundoff). Only assert where the signal is well above noise.
            if abs(v0a) > 1e-10:
                assert int(np.round(np.real(v0b / v0a))) == _spin_parity(m, 0), \
                    f"v_0 parity m={m}"
            if abs(vpa) > 1e-10:
                assert int(np.round(np.real(vpb / vpa))) == _spin_parity(m, 1), \
                    f"v_+ parity m={m}"
            if abs(vma) > 1e-10:
                assert int(np.round(np.real(vmb / vma))) == _spin_parity(m, -1), \
                    f"v_- parity m={m}"

    def test_parity_formula_analytic_consistency(self):
        """The (-1)^|m-s| formula is internally consistent with the
        sin^|m-s| leading-power argument for all m (incl. high m where a
        polynomial test field carries no Fourier signal)."""
        for m in range(0, 12):
            for s in (-1, 0, 1):
                # parity = (-1)^(leading sin power) = (-1)^|m-s|
                assert _spin_parity(m, s) == (-1) ** abs(m - s)
            # v_+ and v_- share parity (|m-1|,|m+1| differ by even number)
            assert _spin_parity(m, 1) == _spin_parity(m, -1)
            # ...and it is opposite to v_0 for m≥1
            if m >= 1:
                assert _spin_parity(m, 1) == -_spin_parity(m, 0)

    def test_m1_vplus_finite_at_pole(self):
        """For m=1, v_+ stays finite at the pole (leading power sin^0)
        while v_r vanishes — the case Bekki's all-zero clamp gets wrong."""
        def field(x, y, z):
            return (x * z + 0.3 * y**2 + 0.5,
                    y * z - 0.2 * x**2 + 0.1,
                    1.0 + 0.4 * x * y + z**2)

        def spin_at(theta, m=1):
            Nph = 256
            ph = np.linspace(0, 2 * np.pi, Nph, endpoint=False)
            x = np.sin(theta) * np.cos(ph)
            y = np.sin(theta) * np.sin(ph)
            z = np.cos(theta) * np.ones_like(ph)
            Vx, Vy, Vz = field(x, y, z)
            vr = (np.sin(theta) * np.cos(ph) * Vx
                  + np.sin(theta) * np.sin(ph) * Vy + np.cos(theta) * Vz)
            vt = (np.cos(theta) * np.cos(ph) * Vx
                  + np.cos(theta) * np.sin(ph) * Vy - np.sin(theta) * Vz)
            vp = -np.sin(ph) * Vx + np.cos(ph) * Vy
            fm = lambda f: np.sum(f * np.exp(-1j * m * ph)) / Nph
            return abs(fm(vr)), abs((fm(vt) - 1j * fm(vp)) / np.sqrt(2))

        vr_small, vp_far = spin_at(0.2)
        vr_tiny, vp_near = spin_at(0.02)
        # v_r shrinks ~10x as θ shrinks 10x (∝ sinθ); v_+ stays ~constant.
        assert vr_tiny < 0.2 * vr_small, "v_r should vanish ∝ sinθ at pole"
        assert abs(vp_near - vp_far) / vp_far < 0.05, (
            "v_+ should stay finite (≈const) approaching the pole")


class TestCGridResolvesBothDefects:
    """The properly-assembled 2D staggered (Arakawa-C) grid should
    recover physical inertial modes that are simultaneously
    checkerboard-free AND undamped (purely imaginary spectrum).

    Diagnostics use the ROBUST criteria — convergence to the analytic
    eigenvalue and machine-precision Re(λ) — not the colocated-calibrated
    eigenvector-alternation score, which misreads physical face-to-face
    structure on a staggered grid as 'checkerboard'.
    """

    def test_recovers_analytic_mode_undamped(self):
        """The ℓ=m inertial mode at i/(m+1) is recovered to machine-
        precision Re(λ)≈0 (undamped) and the correct frequency."""
        m = 2
        grid = make_shell_grid(N_r=48, N_theta=48, r_inner=0.3, r_outer=1.0)
        K, M, _, _ = build_primitive_inertial_evp_cgrid(grid, m=m)
        target = 1j / (m + 1)
        lam = solve_inertial_evp(K, M, sigma=target + 0.005, k=8)
        best = lam[np.argmin(np.abs(lam - target))]
        rel = abs(best.imag - target.imag) / target.imag
        assert rel < 0.02, f"frequency off: {best.imag:.5f}i vs {target.imag:.5f}i"
        assert abs(best.real) < 1e-9, (
            f"mode is spuriously damped: Re(λ)={best.real:.2e} (should be ~0)")

    def test_mode_converges_with_resolution(self):
        """The physical mode converges toward the analytic value as the
        grid refines — the hallmark of a genuine (non-spurious) mode."""
        m = 2
        target = 1j / (m + 1)
        errs = []
        for N in (24, 48):
            grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
            K, M, _, _ = build_primitive_inertial_evp_cgrid(grid, m=m)
            lam = solve_inertial_evp(K, M, sigma=target + 0.005, k=8)
            best = lam[np.argmin(np.abs(lam - target))]
            errs.append(abs(best.imag - target.imag) / target.imag)
        assert errs[1] < errs[0], (
            f"mode did not converge: errs {errs}")

    def test_spectrum_is_essentially_undamped(self):
        """Most modes near a physical target are machine-precision
        undamped, and the worst spurious |Re| is small and shrinks with
        resolution — i.e. no σ-tracking checkerboard cluster."""
        m = 2
        worst = []
        for N in (24, 48):
            grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
            K, M, _, _ = build_primitive_inertial_evp_cgrid(grid, m=m)
            lam = solve_inertial_evp(K, M, sigma=0.001 + 0.0667j, k=20)
            worst.append(np.max(np.abs(lam.real)))
            n_undamped = np.sum(np.abs(lam.real) < 1e-9)
            assert n_undamped >= len(lam) // 2, (
                f"N={N}: only {n_undamped}/{len(lam)} undamped")
        # Worst spurious damping shrinks with refinement (degeneracy
        # splitting, not a fixed checkerboard floor).
        assert worst[1] < worst[0], f"spurious |Re| did not shrink: {worst}"


class TestRadialStaggeringRemovesCheckerboard:
    """The radially-staggered formulation should eliminate the
    checkerboard cluster that swamps the colocated spectrum."""

    def _max_alt_per_mode(self, K, M, grid, sigma, k=20):
        import scipy.sparse.linalg as spla
        N_r, N_theta = grid.N_r, grid.N_theta
        Nv = N_r * N_theta
        KmM = (K - sigma * M).tocsc()
        LU = spla.splu(KmM)
        Lin = spla.LinearOperator(
            K.shape, matvec=lambda x: LU.solve(M @ x), dtype=complex)
        mu, vecs = spla.eigs(Lin, k=k, which="LM", return_eigenvectors=True,
                             maxiter=5000, tol=1e-10)

        def alt(a, axis):
            a = np.real(a)
            if np.max(np.abs(a)) < 1e-14:
                return 0.0
            a = a / np.max(np.abs(a))
            return np.abs(np.diff(a, axis=axis)).mean() / (np.abs(a).mean() + 1e-30)

        out = []
        for i in range(len(mu)):
            vr = vecs[:Nv, i].reshape(N_r, N_theta)
            out.append(max(alt(vr, 0), alt(vr, 1)))
        return np.array(out)

    def test_colocated_is_all_checkerboard_at_physical_target(self):
        """Targeting a physical inertial mode on the colocated grid
        returns only checkerboard modes (alternation ≫ 1)."""
        m = 2
        grid = make_shell_grid(N_r=24, N_theta=48, r_inner=0.3, r_outer=1.0)
        K, M = build_primitive_inertial_evp(grid, m=m, order_r=4, order_theta=6)
        alt = self._max_alt_per_mode(K, M, grid, sigma=0.001 + 0.0667j)
        # Essentially every mode is a node-to-node sawtooth.
        assert np.sum(alt > 1.5) >= 18, (
            f"Expected colocated spectrum to be dominated by checkerboards; "
            f"got alt distribution {np.round(alt, 2)}"
        )

    def test_staggered_has_no_checkerboard(self):
        """The radially-staggered grid returns smooth physical modes
        (alternation < 1) at the same target — no checkerboard."""
        m = 2
        grid = make_shell_grid(N_r=24, N_theta=48, r_inner=0.3, r_outer=1.0)
        K, M = build_primitive_inertial_evp_staggered(
            grid, m=m, order_r=4, order_theta=6)
        alt = self._max_alt_per_mode(K, M, grid, sigma=0.001 + 0.0667j)
        assert np.all(alt < 1.0), (
            f"Staggered grid still has checkerboard modes; "
            f"alt distribution {np.round(alt, 2)}"
        )

    def test_staggered_recovers_physical_eigenvalue(self):
        """A physical inertial eigenvalue (real Re≈0, undamped) is
        cleanly recovered on the staggered grid."""
        m = 2
        grid = make_shell_grid(N_r=24, N_theta=48, r_inner=0.3, r_outer=1.0)
        K, M = build_primitive_inertial_evp_staggered(
            grid, m=m, order_r=4, order_theta=6)
        lam = solve_inertial_evp(K, M, sigma=0.001 + 0.0667j, k=20)
        # Inertial modes are undamped: Re(λ) ≈ 0, Im(λ) > 0.
        undamped = lam[np.abs(lam.real) < 1e-6]
        assert len(undamped) > 0, "No undamped inertial mode found"
        assert np.all(undamped.imag > 0), "Inertial modes should have Im(λ)>0"


# ---------------------------------------------------------------------------
# Tests: demonstrate the checkerboard on the colocated grid
# ---------------------------------------------------------------------------

class TestColocatedCheckerboard:
    """Confirm the colocated primitive formulation produces (a) the
    physical inertial modes and (b) a spurious checkerboard cluster."""

    @pytest.mark.parametrize("m", [1, 2])
    def test_physical_mode_present(self, m):
        """The ℓ=m mode at λ=i/(m+1) is found on the colocated grid."""
        grid = make_shell_grid(N_r=24, N_theta=48, r_inner=0.3, r_outer=1.0)
        K, M = build_primitive_inertial_evp(grid, m=m, order_r=4, order_theta=6)
        target = 1j * m / (m * (m + 1))   # = i/(m+1)
        lam = solve_inertial_evp(K, M, sigma=target + 1e-3, k=30)
        phys, spur = count_checkerboard(lam, m)
        assert len(phys) > 0, "No physical inertial mode recovered"
        best = phys[np.argmin(np.abs(phys - target))]
        rel = abs(best - target) / abs(target)
        assert rel < 0.05, f"m={m}: best phys λ={best:.5f}, target {target:.5f}"

    def test_checkerboard_cluster_exists(self):
        """Targeting the real axis (ω≈0) on the colocated grid pulls in
        a spurious cluster that tracks σ — the checkerboard signature."""
        m = 2
        grid = make_shell_grid(N_r=24, N_theta=48, r_inner=0.3, r_outer=1.0)
        K, M = build_primitive_inertial_evp(grid, m=m, order_r=4, order_theta=6)
        # Target two different real-axis shifts; spurious modes follow σ.
        lam_a = solve_inertial_evp(K, M, sigma=0.05 + 0.0j, k=15)
        lam_b = solve_inertial_evp(K, M, sigma=0.15 + 0.0j, k=15)
        # Most of these should be spurious (not matching analytic).
        _, spur_a = count_checkerboard(lam_a, m)
        _, spur_b = count_checkerboard(lam_b, m)
        # The spurious clusters track the shift: their mean Re follows σ.
        mean_a = np.mean(spur_a.real) if len(spur_a) else np.nan
        mean_b = np.mean(spur_b.real) if len(spur_b) else np.nan
        assert len(spur_a) > 5 and len(spur_b) > 5, (
            "Expected a spurious cluster on the colocated grid"
        )
        # The cluster mean should move toward the (different) shifts.
        assert mean_b > mean_a, (
            f"Spurious cluster did not track σ: mean@0.05={mean_a:.3f}, "
            f"mean@0.15={mean_b:.3f}"
        )
