"""spdee.backends.fd2.ball_origin — the r = 0 treatment of the collocated ball.

Every operator block is assembled in its r^p-multiplied form
(``r²·Lap``, ``r·Div``, ``r√2·Grad``, ...), which is 0 = 0 at r = 0.
The equation masses carry identity placeholders on the i_r = 0 rows, so
those rows of A_total hold the *unscaled* equation at the origin: every
differential builder replaces the i_r = 0 rows of its block by the
operator's r → 0 limit (`install_origin_rows`) and the pointwise,
identity, Dt and source terms land there unscaled by construction.
`CompiledProblem` then reduces the redundant origin unknowns to the
regular degrees of freedom by row replacement, using `origin_shape` and
`ell0_projection_weights` (see the regularity rules).

Regularity rules
----------------
A field is Cartesian-smooth at the origin. For a scalar f with azimuthal
number m the expansion f = Σ_l f_lm(r) Y_l^m with f_lm ~ r^l gives

  scalar, m = 0:   f(0, θ) = f(0), one DOF; ∂_r f(0, θ) = a cos θ (l = 1);
                   f_rr(0, θ) = 2(b sin²θ + c cos²θ) from f = f(0) + a z
                   + b(x² + y²) + c z² + O(r³), so Lap f(0) = 4b + 2c
                   = 3·<f_rr(0, ·)>_{l=0}.
  scalar, m ≠ 0:   f(0, θ) = 0.

For a vector with spherical components ∝ e^{imφ}, the Cartesian
components carry v_z ~ e^{imφ}, v_x ± i v_y ~ e^{i(m±1)φ}, so

  vector, m = 0:   v(0) = c ẑ:  v^0 = c cos θ,  v^± = −c sin θ/√2.
  vector, m = +1:  v(0) = (α/√2)(x̂ + i ŷ), α the amplitude of
                   v_α = (x̂ − i ŷ)·v/√2 (an m = 0 scalar):
                   v^0 = (α/√2) sin θ,  v^- = (α/2)(cos θ + 1),
                   v^+ = (α/2)(cos θ − 1);  m = −1 swaps v^- ↔ v^+.
  vector, |m| ≥ 2: v(0) = 0.

Spin basis: v^± = (v_θ ± i v_φ)/√2, v^0 = v_r, and the shape functions
s_σ(θ) above satisfy Σ_σ s_σ² = 1, so the Cartesian projection that
extracts the single origin DOF is the same combination:
v_z = Σ_σ s_σ v^σ (m = 0) and v_α = Σ_σ s_σ v^σ (|m| = 1).

A rank-2 tensor at the origin is a constant Cartesian matrix T_ij; its
covariant spin components T_{σ_a σ_b}(0, θ, φ) = e_σa · T · e_σb, with
e_0 = r̂ and e_± = (θ̂ ∓ i φ̂)/√2, carry e^{imφ} content for |m| ≤ 2 (the
l = 0, 1, 2 parts of the nine T_ij), so at mode m the nine components
at r = 0 span a space of dimension 3 (m = 0), 2 (|m| = 1), 1 (|m| = 2)
and vanish for |m| ≥ 3 (`tensor_origin_basis`, derived symbolically as
the e^{imφ} coefficient of the nine Cartesian unit tensors and
orthonormalised on the θ grid). Such an unknown keeps one anchor row per
origin DOF and ties every other origin value to the anchors through the
basis.

Operator limits at r = 0
------------------------
Every closed-form operator is assembled from its r^p-multiplied spin
expression Σ_t coef_t(r, θ)·∂^α v^σ_t, polynomial in r. Along a ray a
Cartesian-smooth field is smooth in r, v^σ(r, θ) = Σ_k r^k V_k^σ(θ)
with V_k^σ = ∂_r^k v^σ(0, θ)/k!, so the limit of op(v) at r = 0 is the
coefficient of r^p of the expression after that substitution
(L'Hospital's rule applied symbolically; `lhospital_origin_rows`). It
is a combination of the V_k^σ(θ) and their θ-derivatives, e.g. for
r²·Lap f: 6 V_2 + ∇_H² V_2, which equals 4b + 2c for
V_2 = b sin²θ + c cos²θ. The lower coefficients of r vanish only for
regular fields and never enter. Discretely V_k^σ(θ_j) is the one-sided
k-th derivative at r = 0 over order_r + k nodes (evaluated at the
actual node positions, so non-uniform grids are supported) and the
θ-derivatives use the bundle's stencils. The rows are pointwise in θ;
for a regular field they equal s_σ(θ_j) times the projected value.
Background coefficients b(r, θ) (advection, Dot(bg, tensor)) are
expanded the same way, with B_k(θ_j) taken from the background's grid
values. The composed MHD blocks X·C and C·X inherit the Curl rows, since
X is pointwise. LapH is not Cartesian-regular (f_rr + 2f_r/r and
∇_H²f/r² diverge separately for l = 1 content) and its rows are left as
the angular operator on f(0, ·).

An origin row must involve node-0 stencils of the unknown: a row built
from the interior rows of the same block (extrapolation of
block(r_i)/r_i^p, or a derivative of the r^p-scaled block) is a linear
combination of the interior equations and leaves the origin DOF
undetermined.

Anchor rows
-----------
For an unknown with one origin DOF, `CompiledProblem` keeps one row per
unknown, the l = 0 projection of the Cartesian-projected equation at
r = 0,  Σ_j w_j Σ_σ s_σ(θ_j) · row(σ, i_r = 0, θ_j),  applied to A, to
the mass M and to the source b, so coefficient, sign, pointwise,
identity and Dt terms enter exactly as in the equation. The remaining
rows tie v^σ(0, θ_j) to the anchor value through s_σ(θ_j)/s_anchor(θ_0).
Unknowns that vanish at the origin receive Dirichlet-zero rows.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sps
import sympy as sp

from spdee.backends.fd2.stencils import _lagrange_derivative_weights
from spdee.backends.fd2.term_classify import (
    _decompose_v_expr, ACTION_POINTWISE, ACTION_D_T, ACTION_D_TT,
)
from spdee.ir.fields import r as _r_sym, theta as _t_sym


def _sigma_label(sigma) -> str:
    """Label of a spin index, or of an index pair of a tensor component."""
    if isinstance(sigma, tuple):
        return "".join(_sigma_label(s) for s in sigma)
    return {-1: "m", 0: "0", +1: "p"}[sigma]


def ell0_projection_weights(theta: np.ndarray) -> np.ndarray:
    """Weights w with Σ_j w_j f(θ_j) = <f>_{l=0}, the sphere average of f.

    Solves V^T w = e_0 with the Legendre Vandermonde V[j, l] = P_l(cos θ_j),
    exact for polynomials in cos θ up to degree N_θ − 1 and spectrally
    accurate for smooth axisymmetric fields; the condition number stays
    below 20 on cell-centred grids up to N_θ ≈ 100.
    """
    from scipy.special import eval_legendre
    N = len(theta)
    cos_theta = np.cos(theta)
    V = np.zeros((N, N))
    for ell in range(N):
        V[:, ell] = eval_legendre(ell, cos_theta)
    e0 = np.zeros(N)
    e0[0] = 1.0
    return np.linalg.solve(V.T, e0)


def origin_shape(
    m: int, kind: str, theta: np.ndarray,
) -> Optional[Dict[int, np.ndarray]]:
    """{σ: s_σ(θ_j)} with v^σ(0, θ_j) = c · s_σ(θ_j) for the one origin
    DOF c; {0: 1} for an m = 0 scalar. None when the unknown vanishes at
    the origin (scalar m ≠ 0, vector |m| ≥ 2)."""
    if kind == "scalar":
        return {0: np.ones_like(theta)} if m == 0 else None
    if kind != "vector":
        raise NotImplementedError(
            f"Ball-mode origin regularity is defined for scalar and vector "
            f"unknowns only; got {kind!r}"
        )
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sqrt2 = np.sqrt(2.0)
    if m == 0:
        return {0: cos_t, -1: -sin_t / sqrt2, +1: -sin_t / sqrt2}
    if m == +1:
        return {0: sin_t / sqrt2, -1: (cos_t + 1) / 2, +1: (cos_t - 1) / 2}
    if m == -1:
        return {0: sin_t / sqrt2, -1: (cos_t - 1) / 2, +1: (cos_t + 1) / 2}
    return None


def _tensor_origin_candidates(m: int):
    """The nine covariant spin-component functions of θ, one per
    Cartesian unit tensor E_ij, of the e^{imφ} part of e_σa · E_ij · e_σb:
    {(σ_a, σ_b): SymPy expression in θ}, cached per m."""
    cache = _tensor_origin_candidates.__dict__.setdefault("cache", {})
    if m in cache:
        return cache[m]
    th = _t_sym
    z = sp.Symbol("_z")
    cos_p, sin_p = (z + 1 / z) / 2, (z - 1 / z) / (2 * sp.I)
    r_hat = sp.Matrix([sp.sin(th) * cos_p, sp.sin(th) * sin_p, sp.cos(th)])
    t_hat = sp.Matrix([sp.cos(th) * cos_p, sp.cos(th) * sin_p, -sp.sin(th)])
    p_hat = sp.Matrix([-sin_p, cos_p, 0])
    e = {0: r_hat, +1: (t_hat - sp.I * p_hat) / sp.sqrt(2),
         -1: (t_hat + sp.I * p_hat) / sp.sqrt(2)}
    out = []
    for i in range(3):
        for j in range(3):
            comps = {}
            for a in (-1, 0, +1):
                for b in (-1, 0, +1):
                    poly = sp.expand(e[a][i] * e[b][j] * z ** 4)
                    comps[(a, b)] = sp.simplify(poly.coeff(z, m + 4))
            out.append(comps)
    cache[m] = out
    return out


def tensor_origin_basis(m: int, theta: np.ndarray):
    """Orthonormal basis of the origin values of a tensor unknown at mode
    m: a list of {(σ_a, σ_b): s(θ_j)} dicts (empty when the tensor
    vanishes at r = 0, |m| ≥ 3), orthonormal in the discrete l2 inner
    product over the nine components and the θ nodes."""
    cands = _tensor_origin_candidates(m)
    keys = [(a, b) for a in (-1, 0, +1) for b in (-1, 0, +1)]
    rows = []
    for comps in cands:
        vals = []
        for key in keys:
            fn = sp.lambdify(_t_sym, comps[key], modules="numpy")
            vals.append(np.broadcast_to(np.asarray(fn(theta), dtype=complex), theta.shape))
        rows.append(np.concatenate(vals))
    S = np.array(rows)
    if not np.any(S):
        return []
    _u, sing, vh = np.linalg.svd(S, full_matrices=False)
    keep = sing > 1e-10 * sing[0]
    N_theta = len(theta)
    basis = []
    for vec in vh[keep]:
        basis.append({key: vec[k * N_theta:(k + 1) * N_theta] for k, key in enumerate(keys)})
    return basis


def origin_anchor_sigma(m: int, kind: str) -> int:
    """Spin block hosting the anchor row: σ = 0 for scalars and m = 0
    vectors, the block whose shape function is ≈ 1 at θ_0 for |m| = 1."""
    if kind == "scalar" or m == 0:
        return 0
    return -1 if m == +1 else +1


def _origin_derivative_weights(r: np.ndarray, k: int, order: int) -> np.ndarray:
    """Weights of ∂_r^k(·)(0)/k! over the first order + k nodes (order + k
    one-sided nodes keep formal order `order` for a k-th derivative)."""
    if k == 0:
        return np.array([1.0])
    n = order + k
    if len(r) < n:
        raise ValueError(
            f"Ball origin rows need N_r >= order_r + {k} = {n} radial "
            f"nodes; got N_r = {len(r)}"
        )
    return _lagrange_derivative_weights(r[:n], 0.0, order=k) / math.factorial(k)


def _theta_stencils(bundle):
    """1D (I, D1, D2) on the θ grid at the bundle's order, cached."""
    if not hasattr(bundle, "_origin_theta_stencils"):
        from spdee.backends.fd2.stencils import standard_d1, standard_d2
        theta = bundle.grid.theta
        _, D1 = standard_d1(theta, order=bundle.order_theta)
        _, D2 = standard_d2(theta, order=bundle.order_theta)
        bundle._origin_theta_stencils = {
            ACTION_POINTWISE: sps.identity(len(theta), format="csr"),
            ACTION_D_T: D1.tocsr(),
            ACTION_D_TT: D2.tocsr(),
        }
    return bundle._origin_theta_stencils


def lhospital_origin_rows(
    expr_per_sigma: Dict[int, sp.Expr],
    placeholder_to_id: Dict,
    bundle,
    p_op: int,
    sigmas_in: Sequence[int],
    sigmas_out: Sequence[int],
    bg_values: Optional[Dict] = None,
) -> sps.csr_matrix:
    """The i_r = 0 rows of an operator from its closed-form expressions.

    `expr_per_sigma[σ_out]` is the r^p_op-multiplied spin expression,
    linear in the placeholder functions of `placeholder_to_id`
    ({func: (unknown_name, σ_in)}) and polynomial in r. Each placeholder
    is replaced by Σ_{k≤p+1} r^k V_k^σ(θ), each background placeholder
    in `bg_values` ({applied function: (N_r, N_θ) grid values}) by
    Σ_k r^k B_k(θ), and the coefficient of r^p_op is discretised with
    the one-sided k-th derivative weights at r = 0 and the θ stencils.
    Returns the (n_out·N, n_in·N) matrix whose only nonzero rows are the
    i_r = 0 rows, in the spin orders given.
    """
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    N = N_r * N_theta
    K = p_op + 1
    r_sym, t_sym = _r_sym, _t_sym
    V = {}
    V_to_id = {}
    for sigma in sigmas_in:
        for k in range(K + 1):
            fn = sp.Function(f"_V{k}_{_sigma_label(sigma)}")(t_sym)
            V[(sigma, k)] = fn
            V_to_id[fn.func] = (f"V{k}", sigma)
    func_to_sigma = {func: sig for func, (_, sig) in placeholder_to_id.items()}
    bg_series = {}
    bg_sym = {}
    bg_arrays = {}
    if bg_values:
        for idx, (applied, values) in enumerate(bg_values.items()):
            series = 0
            for k in range(K + 1):
                sym = sp.Symbol(f"_B{k}_{idx}")
                fn = sp.Function(f"_B{k}_{idx}")(t_sym)
                bg_sym[fn] = sym
                w = _origin_derivative_weights(grid.r, k, bundle.order_r)
                bg_arrays[sym] = np.asarray(values, dtype=complex)[:len(w), :].T @ w
                series += r_sym ** k * fn
            bg_series[applied.func] = series
    theta_ops = _theta_stencils(bundle)
    sub_blocks = {
        (so, si): sps.csr_matrix((N, N), dtype=complex)
        for so in sigmas_out for si in sigmas_in
    }
    weights_cache = {}
    for sigma_out in sigmas_out:
        expr = expr_per_sigma[sigma_out]
        for func, sigma in func_to_sigma.items():
            series = sum(r_sym ** k * V[(sigma, k)] for k in range(K + 1))
            expr = expr.replace(func, sp.Lambda((r_sym, t_sym), series))
        for func, series in bg_series.items():
            expr = expr.replace(func, sp.Lambda((r_sym, t_sym), series))
        expr = sp.expand(expr.doit())
        G = sp.expand(expr.coeff(r_sym, p_op))
        if G.has(r_sym):
            raise ValueError(
                f"Origin limit: the r^{p_op} coefficient still depends on r "
                f"(the operator expression is not polynomial in r): {G}"
            )
        for term in _decompose_v_expr(G, V_to_id):
            k = int(term["unknown_name"][1:])
            sigma_in = term["sigma_in"]
            action = term["action"]
            if action not in theta_ops:
                raise NotImplementedError(
                    f"Origin limit: unexpected action {action!r} in {term}"
                )
            if k not in weights_cache:
                weights_cache[k] = _origin_derivative_weights(grid.r, k, bundle.order_r)
            w = weights_cache[k]
            coef = term["coef"]
            for fn, sym in bg_sym.items():
                coef = coef.subs(fn, sym)
            syms = sorted(coef.free_symbols & set(bg_arrays), key=lambda x: x.name)
            fn_c = sp.lambdify((t_sym, *syms), coef, modules="numpy")
            with np.errstate(divide="ignore", invalid="ignore"):
                c = fn_c(grid.theta, *[bg_arrays[sm] for sm in syms])
            c = np.broadcast_to(np.asarray(c, dtype=complex), (N_theta,)).copy()
            if not np.all(np.isfinite(c)):
                raise ValueError(
                    f"Origin limit: coefficient {coef} is not finite on the θ grid"
                )
            C_theta = sps.diags(c, format="csr") @ theta_ops[action]
            W = sps.csr_matrix(
                (w, (np.zeros(len(w), dtype=int), np.arange(len(w)))),
                shape=(N_r, N_r),
            )
            sub_blocks[(sigma_out, sigma_in)] = (
                sub_blocks[(sigma_out, sigma_in)] + sps.kron(C_theta, W, format="csr")
            ).tocsr()
    return sps.bmat(
        [[sub_blocks[(so, si)] for si in sigmas_in] for so in sigmas_out],
        format="csr", dtype=complex,
    )


def install_origin_rows(
    block: sps.spmatrix,
    expr_per_sigma: Dict[int, sp.Expr],
    placeholder_to_id: Dict,
    bundle,
    p_op: int,
    sigmas_in: Sequence[int],
    sigmas_out: Sequence[int],
    bg_values: Optional[Dict] = None,
) -> sps.csr_matrix:
    """`block` with its i_r = 0 rows replaced by `lhospital_origin_rows`
    on a ball grid; unchanged on a shell."""
    grid = bundle.grid
    if not grid.is_ball:
        return block
    N = grid.N_r * grid.N_theta
    d = np.ones(block.shape[0])
    for s in range(len(sigmas_out)):
        for j in range(grid.N_theta):
            d[s * N + j * grid.N_r] = 0.0
    rows = lhospital_origin_rows(
        expr_per_sigma, placeholder_to_id, bundle, p_op,
        sigmas_in, sigmas_out, bg_values,
    )
    return (sps.diags(d, format="csr") @ block + rows).tocsr()
