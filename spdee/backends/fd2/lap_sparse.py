"""spdee.backends.fd2.lap_sparse — sparse builders for r²·Lap(p),
r²·∇_H²(p) and r²·Lap(v).

Each operator is multiplied through by r² to clear its 1/r and 1/r²
factors, then discretized with the bundle's explicit FD stencils. The
result is a sparse block accumulated into A_total for an equation whose
mass is `diag(r²)` (`r2_vector_mass` for vectors); on ball grids the
i_r=0 rows of Lap hold its r→0 limit (ball_origin.py) and those of the
angular ∇_H² stay the angular operator on f(0, ·).

Closed-form r²-multiplied spin formulas:

Scalar:
    r² Lap(f) = r² ∂²_r f + 2r ∂_r f + ∂²_θ f + cot θ · ∂_θ f
              - (m²/sin²θ) f

Horizontal (angular) part:
    r² ∇_H² f = ∂²_θ f + cot θ · ∂_θ f - (m²/sin²θ) f

Vector (one row per output spin):
    r² Lap(v)^0 = r² Lap_scalar(v^0) - 2·v^0
                - √2·(∂_θ + cot θ - m/sin θ) v^-
                - √2·(∂_θ + cot θ + m/sin θ) v^+

    r² Lap(v)^- = r² Lap_scalar(v^-) + √2·(∂_θ + m/sin θ) v^0
                + (1/sin²θ)·(-1 + 2m·cos θ) v^-

    r² Lap(v)^+ = r² Lap_scalar(v^+) + √2·(∂_θ - m/sin θ) v^0
                + (1/sin²θ)·(-1 - 2m·cos θ) v^+
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
    ACTION_POINTWISE, ACTION_D_R, ACTION_D_T, ACTION_D_RR, ACTION_D_TT,
)
from spdee.ir.fields import r as r_sym, theta as t_sym


def _r2_lap_scalar_formula(m: int, f_placeholder: sp.Function) -> sp.Expr:
    """r²·Lap(f) in the spin basis at fixed m."""
    f = f_placeholder
    return sp.expand(
        r_sym**2 * sp.diff(f, r_sym, 2)
        + 2 * r_sym * sp.diff(f, r_sym)
        + sp.diff(f, t_sym, 2)
        + (sp.cos(t_sym) / sp.sin(t_sym)) * sp.diff(f, t_sym)
        - (m**2 / sp.sin(t_sym)**2) * f
    )


def _r2_laph_scalar_formula(m: int, f_placeholder: sp.Function) -> sp.Expr:
    """r²·∇_H²(f): the angular part of r²·Lap(f), no radial derivatives."""
    f = f_placeholder
    return sp.expand(
        sp.diff(f, t_sym, 2)
        + (sp.cos(t_sym) / sp.sin(t_sym)) * sp.diff(f, t_sym)
        - (m**2 / sp.sin(t_sym)**2) * f
    )


def _r2_lap_vector_formulas(
    m: int,
    v_placeholders: Dict[int, sp.Function],
) -> Dict[int, sp.Expr]:
    """{σ_out: r²·Lap(v)^σ_out} in the spin basis at fixed m."""
    vm = v_placeholders[-1]
    v0 = v_placeholders[0]
    vp = v_placeholders[+1]
    sqrt2 = sp.sqrt(2)
    cot_t = sp.cos(t_sym) / sp.sin(t_sym)
    inv_sin = 1 / sp.sin(t_sym)
    inv_sin2 = 1 / sp.sin(t_sym)**2

    def lap_scalar_of(f):
        return _r2_lap_scalar_formula(m, f)

    expr_0 = (
        lap_scalar_of(v0)
        - 2 * v0
        - sqrt2 * (sp.diff(vm, t_sym) + cot_t * vm - m * inv_sin * vm)
        - sqrt2 * (sp.diff(vp, t_sym) + cot_t * vp + m * inv_sin * vp)
    )
    expr_m = (
        lap_scalar_of(vm)
        + sqrt2 * (sp.diff(v0, t_sym) + m * inv_sin * v0)
        + inv_sin2 * (-1 + 2 * m * sp.cos(t_sym)) * vm
    )
    expr_p = (
        lap_scalar_of(vp)
        + sqrt2 * (sp.diff(v0, t_sym) - m * inv_sin * v0)
        + inv_sin2 * (-1 - 2 * m * sp.cos(t_sym)) * vp
    )
    return {-1: sp.expand(expr_m), 0: sp.expand(expr_0), +1: sp.expand(expr_p)}


def _lap_sparse_stencils(bundle: DerivativeBundle):
    """Lifted 2D d1 and d2 stencils in r and θ at the bundle's orders,
    cached on the bundle.
    """
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    if not hasattr(bundle, "_lap_sparse_stencils"):
        from spdee.backends.fd2.stencils import standard_d1, standard_d2
        _, B_1_r_std = standard_d1(grid.r, order=bundle.order_r)
        _, B_1_t_std = standard_d1(grid.theta, order=bundle.order_theta)
        _, B_2_r_std = standard_d2(grid.r, order=bundle.order_r)
        _, B_2_t_std = standard_d2(grid.theta, order=bundle.order_theta)
        bundle._lap_sparse_stencils = {
            "B_1_r_lift": lift_r_op(B_1_r_std, N_theta),
            "B_1_t_lift": lift_theta_op(B_1_t_std, N_r),
            "B_2_r_lift": lift_r_op(B_2_r_std, N_theta),
            "B_2_t_lift": lift_theta_op(B_2_t_std, N_r),
        }
    return bundle._lap_sparse_stencils


def build_r2_lap_scalar_block(
    bundle: DerivativeBundle,
    m: int,
    f_unknown_name: str,
) -> sps.csr_matrix:
    """(N, N) sparse block with diag(r²) · Lap(f) = block · f_state,
    N = N_r·N_theta.
    """
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    block_size = N_r * N_theta
    stencils = _lap_sparse_stencils(bundle)
    f_placeholder = sp.Function(f"{f_unknown_name}_sigma_0")(r_sym, t_sym)
    expr = _r2_lap_scalar_formula(m, f_placeholder)
    placeholder_to_id = {f_placeholder.func: (f_unknown_name, 0)}
    out_block = sps.csr_matrix((block_size, block_size), dtype=complex)
    terms = _decompose_v_expr(expr, placeholder_to_id)
    for term in terms:
        coef_diag = _coef_to_diag(term["coef"], grid)
        action = term["action"]
        if action == ACTION_POINTWISE:
            op = sps.eye(block_size, dtype=complex, format="csr")
        elif action == ACTION_D_R:
            op = stencils["B_1_r_lift"]
        elif action == ACTION_D_T:
            op = stencils["B_1_t_lift"]
        elif action == ACTION_D_RR:
            op = stencils["B_2_r_lift"]
        elif action == ACTION_D_TT:
            op = stencils["B_2_t_lift"]
        else:
            raise NotImplementedError(
                f"r²·Lap shouldn't produce {action!r} term; got {term}"
            )
        contrib = (coef_diag @ op).tocsr()
        out_block = (out_block + contrib).tocsr()
    return install_origin_rows(
        out_block, {0: expr}, placeholder_to_id, bundle, 2, (0,), (0,),
    )


def build_r2_laph_scalar_block(
    bundle: DerivativeBundle,
    m: int,
    f_unknown_name: str,
) -> sps.csr_matrix:
    """(N, N) sparse block with diag(r²) · ∇_H²(f) = block · f_state.

    ∇_H² has no radial derivatives, so the block is the same angular
    operator repeated at every radial node.
    """
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    block_size = N_r * N_theta
    stencils = _lap_sparse_stencils(bundle)
    f_placeholder = sp.Function(f"{f_unknown_name}_sigma_0")(r_sym, t_sym)
    expr = _r2_laph_scalar_formula(m, f_placeholder)
    placeholder_to_id = {f_placeholder.func: (f_unknown_name, 0)}
    out_block = sps.csr_matrix((block_size, block_size), dtype=complex)
    terms = _decompose_v_expr(expr, placeholder_to_id)
    for term in terms:
        coef_diag = _coef_to_diag(term["coef"], grid)
        action = term["action"]
        if action == ACTION_POINTWISE:
            op = sps.eye(block_size, dtype=complex, format="csr")
        elif action == ACTION_D_T:
            op = stencils["B_1_t_lift"]
        elif action == ACTION_D_TT:
            op = stencils["B_2_t_lift"]
        else:
            raise NotImplementedError(
                f"r²·∇_H² shouldn't produce {action!r} term; got {term}"
            )
        contrib = (coef_diag @ op).tocsr()
        out_block = (out_block + contrib).tocsr()
    return out_block


def build_r2_lap_vector_block(
    walker_meta: Dict[str, Any],
    bundle: DerivativeBundle,
    m: int,
) -> sps.csr_matrix:
    """(3N, 3N) sparse block with r2_vector_mass · Lap(v)^σ = block · v_state.

    r²·Lap(v) never produces a mixed ∂_r∂_θ term, and the assembler
    enforces that.
    """
    v_placeholders_dict = walker_meta["v_placeholders"]
    v_unk_name = list(v_placeholders_dict.keys())[0]
    v_funcs = v_placeholders_dict[v_unk_name]
    formulas = _r2_lap_vector_formulas(m, v_funcs)
    placeholder_to_id = {
        v_funcs[s].func: (v_unk_name, s) for s in (-1, 0, +1)
    }
    grid = bundle.grid

    def coef_to_diag(coef_expr):
        return _coef_to_diag(coef_expr, grid)

    return assemble_vector_spin_block(
        formulas, placeholder_to_id, bundle, m, coef_to_diag,
        v_unknown_name=v_unk_name,
        builder_name="build_r2_lap_vector_block",
        allowed_actions=frozenset({
            ACTION_POINTWISE, ACTION_D_R, ACTION_D_T, ACTION_D_RR, ACTION_D_TT,
        }),
        origin_p_op=2,
    )
