"""spdee.backends.fd2.div_tensor_sparse — sparse builder for Div(tensor).

Builds the (3N, 3N) block for `r²·(Div T)^σ` directly from the closed-form
spin-basis expression in v's first and second derivatives, using only
explicit FD stencils. The number of nonzeros per row depends on the
stencil width, not on the grid size.

The equation mass is `r2_vector_mass` (diag(r²) with identity
placeholders at i_r=0), so the relation enforced is

    diag(r²) · (Div T)^σ = build_div_tensor_vector_block(...) · v_state

The closed form comes from substituting the walker's `U_in_v[(σ_a, σ_b)]`
into `per_sigma_expr_in_U[σ]`. Each additive term has the form
`coef(r, θ) · derivative_of_v^{σ_in}` with derivative_of_v ∈
{1, ∂_r, ∂_θ, ∂²_r, ∂²_θ, ∂_r∂_θ} and discretizes to
`diag(coef) · D_action · v_state`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import scipy.sparse as sps
import sympy as sp

from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.spin_block_assembly import assemble_vector_spin_block
from spdee.backends.fd2.term_classify import _coef_to_diag


def closed_form_in_v(
    per_sigma_expr_in_U: Dict[int, sp.Expr],
    U_placeholders: Dict[Tuple[int, int], sp.Function],
    U_in_v: Dict[Tuple[int, int], sp.Expr],
) -> Dict[int, sp.Expr]:
    """Substitute U_in_v[(σ_a, σ_b)] into per_sigma_expr_in_U[σ_out] and
    return {σ_out: expression for r²·(Div T)^σ_out in v's derivatives}.
    """
    out = {}
    for sigma_out, expr in per_sigma_expr_in_U.items():
        e = expr
        for key, U_func in U_placeholders.items():
            u_expr = U_in_v[key]
            e = e.replace(U_func.func, sp.Lambda(U_func.args, u_expr))
        out[sigma_out] = sp.expand(e.doit())
    return out


def build_div_tensor_vector_block(
    walker_meta: "Dict[str, object]",
    bundle: DerivativeBundle,
    m: int,
) -> sps.csr_matrix:
    """(3N, 3N) sparse block with diag(r²) · (Div T)^σ = block · v_state,
    in spin order (-1, 0, +1). Accumulated into A_total for a
    TASK_DIV_TENSOR_VECTOR with `r2_vector_mass` as the equation mass.
    """
    per_sigma_expr_in_U = walker_meta["per_sigma_expr_in_U"]
    U_in_v = walker_meta["U_in_v"]
    U_placeholders = walker_meta["U_placeholders"]
    v_placeholders = walker_meta["v_placeholders"]
    expr_in_v = closed_form_in_v(
        per_sigma_expr_in_U, U_placeholders, U_in_v,
    )
    v_unk_name = list(v_placeholders.keys())[0]
    v_funcs = v_placeholders[v_unk_name]
    v_placeholder_to_id = {
        v_funcs[s].func: (v_unk_name, s) for s in (-1, 0, +1)
    }
    grid = bundle.grid

    def coef_to_diag(coef_expr):
        return _coef_to_diag(coef_expr, grid)

    return assemble_vector_spin_block(
        expr_in_v, v_placeholder_to_id, bundle, m, coef_to_diag,
        v_unknown_name=v_unk_name,
        builder_name="build_div_tensor_vector_block",
        origin_p_op=2,
    )
