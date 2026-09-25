"""spdee.backends.fd2.curl_cross_sparse — sparse builders for Curl(v),
Curl(Curl(v)), Cross(b, v) and the composed MHD blocks Cross(Curl(v), b),
Curl(Cross(v, b)).

Curl is first order and Cross is pointwise. Both produce closed-form
spin-basis expressions — `r·(Curl v)^σ` and `Cross(b, v)^σ` — that are
classified into pointwise / ∂_r / ∂_θ actions and discretized with the
bundle's FD stencils into (3N, 3N) blocks for A_total. Curl(Curl(v)) is
the symbolic composition of the Curl formula with itself, multiplied
through by r² (second order, mixed ∂_r∂_θ included).

Equation mass:
  - Curl: `r1_vector_mass` (diag(r) with i_r=0 placeholders), which
    absorbs the 1/r of the Curl formula.
  - Curl(Curl): `r2_vector_mass`, as for Lap and Grad(Div).
  - Cross alone: none (identity). In an equation with Curl or Lap the
    mass is set by that operator and the dispatcher applies it on the
    left of the pointwise Cross block.

Entry points:
  - `build_curl_vector_block(walker_meta, bundle, m)`:
    diag(r) · (Curl v)^σ = block · v.
  - `build_r2_curl_curl_vector_block(walker_meta, bundle, m)`:
    diag(r²) · (Curl Curl v)^σ = block · v.
  - `build_cross_vector_block(b_spin, bundle, m, side)`:
    Cross(b, v)^σ = block · v for side='left', Cross(v, b) = -Cross(b, v)
    for side='right'.
  - `build_cross_curl_unk_bg_block`, `build_curl_cross_unk_bg_block`:
    matrix products of the two elementary blocks (see below).
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import sympy as sp
import scipy.sparse as sps

from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.operators_core import diag_from_2d_array
from spdee.backends.fd2.spin_block_assembly import assemble_vector_spin_block
from spdee.backends.fd2.term_classify import (
    _coef_to_diag, ACTION_POINTWISE, ACTION_D_R, ACTION_D_T,
)
from spdee.ir.fields import r as r_sym, theta as t_sym


def _curl_spin_formulas(m: int, v_placeholders: Dict[int, sp.Function]):
    """{σ_out: r·(Curl v)^σ_out} in terms of the v placeholder Functions.

    From the physical-basis curl in spherical coordinates with
    v_r = v^0, v_θ = (v^+ + v^-)/√2, v_φ = i(v^- - v^+)/√2 and ∂_φ → i·m:
      r·(Curl v)^- = i·(√2·m/(2 sin θ)·v^0 - r·∂_r v^- - v^- + (√2/2)·∂_θ v^0)
      r·(Curl v)^0 = i·(-m·(v^- + v^+)/(√2 sin θ)
                       + (∂_θ(v^- - v^+))/√2
                       + cot(θ)·(v^- - v^+)/√2)
      r·(Curl v)^+ = i·(√2·m/(2 sin θ)·v^0 + r·∂_r v^+ + v^+ - (√2/2)·∂_θ v^0)

    After the factor r no 1/r remains; 1/sin θ is finite on the
    cell-centered θ grid.
    """
    vm = v_placeholders[-1]
    v0 = v_placeholders[0]
    vp = v_placeholders[+1]
    I = sp.I
    sqrt2 = sp.sqrt(2)
    cot_t = sp.cos(t_sym) / sp.sin(t_sym)
    inv_sin = 1 / sp.sin(t_sym)
    curl_m = I * (
        sqrt2 * m / 2 * inv_sin * v0
        - r_sym * sp.diff(vm, r_sym)
        - vm
        + sqrt2 / 2 * sp.diff(v0, t_sym)
    )
    curl_0 = I * (
        - m / (sqrt2 * sp.sin(t_sym)) * (vm + vp)
        + (sp.diff(vm, t_sym) - sp.diff(vp, t_sym)) / sqrt2
        + cot_t * (vm - vp) / sqrt2
    )
    curl_p = I * (
        sqrt2 * m / 2 * inv_sin * v0
        + r_sym * sp.diff(vp, r_sym)
        + vp
        - sqrt2 / 2 * sp.diff(v0, t_sym)
    )
    return {-1: sp.expand(curl_m), 0: sp.expand(curl_0), +1: sp.expand(curl_p)}


def build_curl_vector_block(
    walker_meta: Dict[str, Any],
    bundle: DerivativeBundle,
    m: int,
) -> sps.csr_matrix:
    """(3N, 3N) sparse block with diag(r) · (Curl v)^σ = block · v.

    Curl is first order, so only pointwise / ∂_r / ∂_θ actions are
    allowed and the assembler enforces that.
    """
    v_placeholders = walker_meta["v_placeholders"]
    v_unk_name = list(v_placeholders.keys())[0]
    v_funcs = v_placeholders[v_unk_name]
    formulas = _curl_spin_formulas(m, v_funcs)
    v_placeholder_to_id = {
        v_funcs[s].func: (v_unk_name, s) for s in (-1, 0, +1)
    }
    grid = bundle.grid

    def coef_to_diag(coef_expr):
        return _coef_to_diag(coef_expr, grid)

    return assemble_vector_spin_block(
        formulas, v_placeholder_to_id, bundle, m, coef_to_diag,
        v_unknown_name=v_unk_name,
        builder_name="build_curl_vector_block",
        allowed_actions=frozenset({ACTION_POINTWISE, ACTION_D_R, ACTION_D_T}),
        origin_p_op=1,
    )


def _r2_curl_curl_spin_formulas(m: int, v_placeholders: Dict[int, sp.Function]):
    """{σ_out: r²·(Curl Curl v)^σ_out} in terms of the v placeholders.

    With W^σ = r·(Curl v)^σ from `_curl_spin_formulas`, the outer curl
    of w = W/r gives r²·(Curl w)^σ = r·[r·Curl(W/r)]^σ, which is
    polynomial in r; symbolically it equals r²·Grad(Div v) − r²·Lap(v)
    term by term, so the assembled block is the difference of the
    Grad(Div) and Lap blocks entrywise, origin rows included.
    """
    W = _curl_spin_formulas(m, v_placeholders)
    w = {
        s: sp.Function(f"_curl_curl_w_sigma_{name}")(r_sym, t_sym)
        for s, name in [(-1, "m"), (0, "0"), (+1, "p")]
    }
    outer = _curl_spin_formulas(m, w)
    out = {}
    for s_out in (-1, 0, +1):
        expr = outer[s_out]
        for s_in in (-1, 0, +1):
            expr = expr.replace(
                w[s_in].func, sp.Lambda((r_sym, t_sym), W[s_in] / r_sym),
            )
        out[s_out] = sp.expand(r_sym * expr.doit())
    return out


def build_r2_curl_curl_vector_block(
    walker_meta: Dict[str, Any],
    bundle: DerivativeBundle,
    m: int,
) -> sps.csr_matrix:
    """(3N, 3N) sparse block with r2_vector_mass · (Curl Curl v)^σ = block · v.

    Second order with a mixed ∂_r∂_θ term (the product stencil
    B_1_r ∘ B_1_t, as in Grad(Div)); on a ball the i_r=0 rows hold the
    r→0 limit from the symbolic L'Hospital rule.
    """
    v_placeholders = walker_meta["v_placeholders"]
    v_unk_name = list(v_placeholders.keys())[0]
    v_funcs = v_placeholders[v_unk_name]
    formulas = _r2_curl_curl_spin_formulas(m, v_funcs)
    v_placeholder_to_id = {
        v_funcs[s].func: (v_unk_name, s) for s in (-1, 0, +1)
    }
    grid = bundle.grid

    def coef_to_diag(coef_expr):
        return _coef_to_diag(coef_expr, grid)

    return assemble_vector_spin_block(
        formulas, v_placeholder_to_id, bundle, m, coef_to_diag,
        v_unknown_name=v_unk_name,
        builder_name="build_r2_curl_curl_vector_block",
        origin_p_op=2,
    )


def build_cross_vector_block(
    b_spin_components: Dict[int, np.ndarray],
    bundle: DerivativeBundle,
    m: int,
    side: str = "left",
) -> sps.csr_matrix:
    """(3N, 3N) pointwise block for Cross(b, v) (side='left') or
    Cross(v, b) = -Cross(b, v) (side='right') acting on the vector
    state v.

    In the spin basis
      Cross(b, v)^- = i·(-b^0 · v^- + b^- · v^0)
      Cross(b, v)^0 = i·(-b^- · v^+ + b^+ · v^-)
      Cross(b, v)^+ = i·(+b^0 · v^+ - b^+ · v^0)
    so each (σ_out, σ_in) coupling is a diagonal built from one spin
    component of b on the grid.

    Parameters
    ----------
    b_spin_components : dict {σ: (N_r, N_theta) ndarray}
        Background vector b in the spin basis on the grid.
    bundle : DerivativeBundle
        Provides the grid dimensions; no stencils are used.
    m : int
        Azimuthal mode; unused (b's φ-dependence is already in its
        spin components).
    side : 'left' or 'right'
    """
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    block = N_r * N_theta
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
    sign = +1 if side == "left" else -1
    I = 1j
    bm = b_spin_components[-1]
    b0 = b_spin_components[0]
    bp = b_spin_components[+1]
    zero_2d = np.zeros((N_r, N_theta), dtype=complex)
    couplings = {
        (-1, -1): I * (-b0),
        (-1,  0): I * (+bm),
        (-1, +1): zero_2d,
        ( 0, -1): I * (+bp),
        ( 0,  0): zero_2d,
        ( 0, +1): I * (-bm),
        (+1, -1): zero_2d,
        (+1,  0): I * (-bp),
        (+1, +1): I * (+b0),
    }
    sub_blocks: Dict[tuple, sps.csr_matrix] = {}
    for key, diag_2d in couplings.items():
        if np.all(diag_2d == 0):
            sub_blocks[key] = sps.csr_matrix((block, block), dtype=complex)
        else:
            sub_blocks[key] = (sign * diag_from_2d_array(diag_2d)).tocsr()
    return sps.bmat([
        [sub_blocks[(-1, -1)], sub_blocks[(-1,  0)], sub_blocks[(-1, +1)]],
        [sub_blocks[( 0, -1)], sub_blocks[( 0,  0)], sub_blocks[( 0, +1)]],
        [sub_blocks[(+1, -1)], sub_blocks[(+1,  0)], sub_blocks[(+1, +1)]],
    ], format="csr", dtype=complex)


# Composed MHD blocks. With C the Curl block (diag(r)·Curl) and X the
# pointwise Cross block on a background spin field, X commutes with
# diag(r), so
#
#   r · Cross(Curl(B), B₀)^σ = X · (r · Curl(B))^σ = (X @ C) · B
#   r · Curl(Cross(v, B₀))^σ = C · Cross(v, B₀)^σ  = (C @ X) · v
#
# Both products have r-power 1 and use `r1_vector_mass`, so the
# dispatcher emits them straight into A_total without an auxiliary
# unknown for the inner operator. On a ball the i_r=0 rows of C hold
# the r→0 limit of Curl, and since X is pointwise the products inherit
# them: X(0)·Curl-limit for X @ C, and the Curl limit of the regular
# field u = Cross(v, B₀) for C @ X.


def _make_curl_walker_meta(field_name: str = "_aux_unk") -> Dict[str, Any]:
    """The walker_meta dict `build_curl_vector_block` expects, with
    placeholder Functions for the three spin components. The field
    name is only a marker; the block does not depend on it.
    """
    placeholders = {
        s: sp.Function(f"{field_name}_sigma_{name}")(r_sym, t_sym)
        for s, name in [(-1, "m"), (0, "0"), (+1, "p")]
    }
    return {"v_placeholders": {field_name: placeholders}}


def build_cross_curl_unk_bg_block(
    bg_field,
    bundle,
    grid,
    m: int,
    side: str = "left",
) -> sps.csr_matrix:
    """(3N, 3N) block X @ C for ``Cross(Curl(unknown), bg)`` (side='right')
    or ``Cross(bg, Curl(unknown))`` (side='left'); the two differ by sign.

    Parameters
    ----------
    bg_field : VectorField
        Background vector field.
    bundle : DerivativeBundle
    grid : Grid
        Mesh on which the background is evaluated.
    m : int
        Azimuthal mode.
    side : 'left' or 'right', default 'left'
        Argument ordering of the outer Cross.
    """
    from spdee.backends.common.field_eval import (
        evaluate_vector_background_spin_2d,
    )
    R, T = grid.mesh()
    b_spin = evaluate_vector_background_spin_2d(bg_field, R, T)
    C = build_curl_vector_block(_make_curl_walker_meta(), bundle, m)
    X = build_cross_vector_block(b_spin, bundle, m, side=side)
    return (X @ C).tocsr()


def build_curl_cross_unk_bg_block(
    bg_field,
    bundle,
    grid,
    m: int,
    inner_side: str = "right",
) -> sps.csr_matrix:
    """(3N, 3N) block C @ X for ``Curl(Cross(unknown, bg))``
    (inner_side='right') or ``Curl(Cross(bg, unknown))``
    (inner_side='left'); the two differ by sign.

    Parameters
    ----------
    bg_field : VectorField
        Background vector field inside the Cross.
    bundle : DerivativeBundle
    grid : Grid
    m : int
    inner_side : 'left' or 'right', default 'right'
        Argument ordering of the inner Cross.
    """
    from spdee.backends.common.field_eval import (
        evaluate_vector_background_spin_2d,
    )
    R, T = grid.mesh()
    b_spin = evaluate_vector_background_spin_2d(bg_field, R, T)
    C = build_curl_vector_block(_make_curl_walker_meta(), bundle, m)
    X = build_cross_vector_block(b_spin, bundle, m, side=inner_side)
    return (C @ X).tocsr()


def build_cross_curl_curl_unk_bg_block(
    bg_field,
    bundle,
    grid,
    m: int,
    side: str = "right",
) -> sps.csr_matrix:
    """(3N, 3N) block X @ CC for ``Cross(Curl(Curl(A)), bg)`` (side='right')
    or ``Cross(bg, Curl(Curl(A)))`` (side='left'), the Lorentz force of
    the vector potential with j = Curl(Curl(A)). X is pointwise and
    commutes with diag(r²), so the product carries the r² Curl(Curl)
    mass and inherits its origin rows.
    """
    from spdee.backends.common.field_eval import (
        evaluate_vector_background_spin_2d,
    )
    R, T = grid.mesh()
    b_spin = evaluate_vector_background_spin_2d(bg_field, R, T)
    CC = build_r2_curl_curl_vector_block(_make_curl_walker_meta(), bundle, m)
    X = build_cross_vector_block(b_spin, bundle, m, side=side)
    return (X @ CC).tocsr()
