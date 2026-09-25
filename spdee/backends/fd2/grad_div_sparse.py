"""spdee.backends.fd2.grad_div_sparse — sparse builders for Grad(p),
Div(v) and Grad(Div(v)), with p a scalar unknown and v a vector unknown.

Closed-form spin-basis formulas, multiplied through by the power of r
that clears the 1/r factors:

  Grad(p): scalar -> vector, multiplied by r·√2
    (r·√2)·(Grad p)^0 = r·√2 · ∂_r p
    (r·√2)·(Grad p)^- = m·p/sin θ + ∂_θ p
    (r·√2)·(Grad p)^+ = -m·p/sin θ + ∂_θ p

  Div(v):  vector -> scalar, multiplied by r
    r · Div(v) = r · ∂_r v^0 + 2 · v^0
               + (1/√2) · (∂_θ v^- + cot θ · v^- - m·v^-/sin θ)
               + (1/√2) · (∂_θ v^+ + cot θ · v^+ + m·v^+/sin θ)

  Grad(Div(v)): vector -> vector, multiplied by r² (see
    `_r2_grad_div_vector_formulas`).

Equation masses:
  - Grad's equation (vector): diag(r·√2) per spin block
    (`r_sqrt2_vector_mass`).
  - Div's equation (scalar): diag(r) (`r1_scalar_mass`).
  - Grad(Div)'s equation (vector): diag(r²) per spin block
    (`r2_vector_mass`).
All carry i_r=0 identity placeholders on ball grids, where the blocks'
i_r=0 rows hold the operators' r→0 limits (ball_origin.py).
"""
from __future__ import annotations

from typing import Dict, Any

import sympy as sp
import scipy.sparse as sps

from spdee.backends.fd2.ball_origin import install_origin_rows
from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.operators_core import lift_r_op, lift_theta_op
from spdee.backends.fd2.spin_block_assembly import assemble_vector_spin_block
from spdee.backends.fd2.term_classify import (
    _decompose_v_expr, _coef_to_diag,
    ACTION_POINTWISE, ACTION_D_R, ACTION_D_T,
)
from spdee.ir.fields import r as r_sym, theta as t_sym


def _grad_scalar_spin_formulas(m: int, p_placeholder: sp.Function):
    """{σ_out: (r·√2)·(Grad p)^σ_out} in terms of the placeholder p.

    The factor r·√2 removes the 1/(r·√2) prefactor of the σ=±1 rows and
    scales the σ=0 row.
    """
    p = p_placeholder
    sqrt2 = sp.sqrt(2)
    inv_sin = 1 / sp.sin(t_sym)
    g0 = r_sym * sqrt2 * sp.diff(p, r_sym)
    gm = m * inv_sin * p + sp.diff(p, t_sym)
    gp = -m * inv_sin * p + sp.diff(p, t_sym)
    return {-1: sp.expand(gm), 0: sp.expand(g0), +1: sp.expand(gp)}


def _div_vector_spin_formula(m: int, v_placeholders: Dict[int, sp.Function]):
    """r·Div(v) in terms of v's spin placeholders.

    From
      Div(v) = (1/r²) ∂_r(r² v_r) + (1/(r sin θ)) ∂_θ(sin θ · v_θ)
             + (1/(r sin θ)) · i·m · v_φ
    multiplied by r:
      r·Div(v) = r·∂_r v^0 + 2·v^0 + ∂_θ v_θ + cot θ · v_θ + (i·m/sin θ) v_φ
    with v_θ = (v^+ + v^-)/√2 and v_φ = i·(v^- - v^+)/√2.
    """
    vm = v_placeholders[-1]
    v0 = v_placeholders[0]
    vp = v_placeholders[+1]
    I = sp.I
    sqrt2 = sp.sqrt(2)
    cot_t = sp.cos(t_sym) / sp.sin(t_sym)
    inv_sin = 1 / sp.sin(t_sym)
    v_theta = (vp + vm) / sqrt2
    v_phi = I * (vm - vp) / sqrt2
    expr = (
        r_sym * sp.diff(v0, r_sym) + 2 * v0
        + sp.diff(v_theta, t_sym) + cot_t * v_theta
        + I * m * inv_sin * v_phi
    )
    return sp.expand(expr)


def _grad_div_stencils(bundle: DerivativeBundle):
    """Lifted 2D d1 stencils in r and θ at the bundle's orders, cached
    on the bundle.
    """
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    if not hasattr(bundle, "_grad_div_sparse_stencils"):
        from spdee.backends.fd2.stencils import standard_d1
        _, B_1_r_std = standard_d1(grid.r, order=bundle.order_r)
        _, B_1_t_std = standard_d1(grid.theta, order=bundle.order_theta)
        bundle._grad_div_sparse_stencils = {
            "B_1_r_lift": lift_r_op(B_1_r_std, N_theta),
            "B_1_t_lift": lift_theta_op(B_1_t_std, N_r),
        }
    return bundle._grad_div_sparse_stencils


def build_grad_scalar_block(
    bundle: DerivativeBundle,
    m: int,
    p_unknown_name: str,
) -> sps.csr_matrix:
    """(3N, N) sparse block with diag(r·√2) · (Grad p)^σ = block · p_state.

    Rows are the three vector spin blocks in order (-1, 0, +1); the
    dispatcher places the block at (vector_equation_rows, p_columns).
    """
    p_placeholder = sp.Function(f"{p_unknown_name}_sigma_0")(r_sym, t_sym)
    formulas = _grad_scalar_spin_formulas(m, p_placeholder)
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    block = N_r * N_theta
    stencils = _grad_div_stencils(bundle)
    # The scalar is treated as the σ=0 component of a degenerate vector
    # so the term classifier can be reused.
    placeholder_to_id = {p_placeholder.func: (p_unknown_name, 0)}
    sub_blocks = {}
    for sigma_out, expr in formulas.items():
        terms = _decompose_v_expr(expr, placeholder_to_id)
        sub_block = sps.csr_matrix((block, block), dtype=complex)
        for term in terms:
            coef_diag = _coef_to_diag(term["coef"], grid)
            action = term["action"]
            if action == ACTION_POINTWISE:
                op = sps.eye(block, dtype=complex, format="csr")
            elif action == ACTION_D_R:
                op = stencils["B_1_r_lift"]
            elif action == ACTION_D_T:
                op = stencils["B_1_t_lift"]
            else:
                raise NotImplementedError(
                    f"Grad shouldn't produce {action!r} term; got term={term}"
                )
            contrib = (coef_diag @ op).tocsr()
            sub_block = (sub_block + contrib).tocsr()
        sub_blocks[sigma_out] = sub_block
    op_block = sps.vstack(
        [sub_blocks[-1], sub_blocks[0], sub_blocks[+1]],
        format="csr",
    ).astype(complex)
    return install_origin_rows(
        op_block, formulas, placeholder_to_id, bundle, 1, (0,), (-1, 0, +1),
    )


def build_div_vector_scalar_block(
    walker_meta: Dict[str, Any],
    bundle: DerivativeBundle,
    m: int,
) -> sps.csr_matrix:
    """(N, 3N) sparse block with diag(r) · Div(v) = block · v_state.

    Columns are the three vector spin blocks in order (-1, 0, +1); the
    dispatcher places the block at (scalar_equation_rows, v_columns).
    """
    v_placeholders_dict = walker_meta["v_placeholders"]
    v_unk_name = list(v_placeholders_dict.keys())[0]
    v_funcs = v_placeholders_dict[v_unk_name]
    expr = _div_vector_spin_formula(m, v_funcs)
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    block = N_r * N_theta
    stencils = _grad_div_stencils(bundle)
    placeholder_to_id = {
        v_funcs[s].func: (v_unk_name, s) for s in (-1, 0, +1)
    }
    sub_blocks: Dict[int, sps.csr_matrix] = {
        s: sps.csr_matrix((block, block), dtype=complex)
        for s in (-1, 0, +1)
    }
    terms = _decompose_v_expr(expr, placeholder_to_id)
    for term in terms:
        sigma_in = term["sigma_in"]
        coef_diag = _coef_to_diag(term["coef"], grid)
        action = term["action"]
        if action == ACTION_POINTWISE:
            op = sps.eye(block, dtype=complex, format="csr")
        elif action == ACTION_D_R:
            op = stencils["B_1_r_lift"]
        elif action == ACTION_D_T:
            op = stencils["B_1_t_lift"]
        else:
            raise NotImplementedError(
                f"Div(vector) shouldn't produce {action!r} term; got term={term}"
            )
        contrib = (coef_diag @ op).tocsr()
        sub_blocks[sigma_in] = (sub_blocks[sigma_in] + contrib).tocsr()
    op_block = sps.hstack(
        [sub_blocks[-1], sub_blocks[0], sub_blocks[+1]],
        format="csr",
    ).astype(complex)
    return install_origin_rows(
        op_block, {0: expr}, placeholder_to_id, bundle, 1, (-1, 0, +1), (0,),
    )


def _r2_grad_div_vector_formulas(m: int, v_placeholders: Dict[int, sp.Function]):
    """{σ_out: r²·(Grad(Div(v)))^σ_out}.

    With D_r ≡ r·Div(v) from `_div_vector_spin_formula` and
    Div(v) = D_r/r, the Grad formulas give
        r²·(Grad Div v)^0 = r²·∂_r(D_r/r) = r·∂_r D_r − D_r
        r²·(Grad Div v)^- = (1/√2)·(∂_θ + m/sin θ)·D_r
        r²·(Grad Div v)^+ = (1/√2)·(∂_θ − m/sin θ)·D_r
    The θ-derivatives of D_r introduce ∂_r∂_θ and ∂²_θ terms in the
    σ=± rows.
    """
    sqrt2 = sp.sqrt(2)
    inv_sin = 1 / sp.sin(t_sym)
    D_r = _div_vector_spin_formula(m, v_placeholders)
    f0 = r_sym * sp.diff(D_r, r_sym) - D_r
    fm = (sp.diff(D_r, t_sym) + m * inv_sin * D_r) / sqrt2
    fp = (sp.diff(D_r, t_sym) - m * inv_sin * D_r) / sqrt2
    return {-1: sp.expand(fm), 0: sp.expand(f0), +1: sp.expand(fp)}


def build_r2_grad_div_vector_block(
    walker_meta: Dict[str, Any],
    bundle: DerivativeBundle,
    m: int,
) -> sps.csr_matrix:
    """(3N, 3N) sparse block with r2_vector_mass · Grad(Div(v))^σ = block · v_state.

    The mixed ∂_r∂_θ stencil is the product B_1_r ∘ B_1_t. Dispatching
    the composed operator directly avoids an auxiliary scalar unknown
    D := Div(v), which would add N_r·N_θ rows and columns to the state.
    """
    v_placeholders_dict = walker_meta["v_placeholders"]
    v_unk_name = list(v_placeholders_dict.keys())[0]
    v_funcs = v_placeholders_dict[v_unk_name]
    formulas = _r2_grad_div_vector_formulas(m, v_funcs)
    grid = bundle.grid
    placeholder_to_id = {
        v_funcs[s].func: (v_unk_name, s) for s in (-1, 0, +1)
    }

    def coef_to_diag(coef_expr):
        return _coef_to_diag(coef_expr, grid)

    return assemble_vector_spin_block(
        formulas, placeholder_to_id, bundle, m, coef_to_diag,
        v_unknown_name=v_unk_name,
        builder_name="build_r2_grad_div_vector_block",
        origin_p_op=2,
    )
