"""spdee.backends.fd2.spin_block_assembly — shared spin-block assembler.

Every vector→vector operator builder follows one recipe once its
closed-form spin formulas are available:

  1. classify each per-output-spin SymPy expression into additive terms
     `coef(r, θ) · action(v^{σ_in})`, action ∈ {pointwise, ∂_r, ∂_θ,
     ∂²_r, ∂²_θ, ∂_r∂_θ};
  2. turn each term's coefficient into a sparse diagonal on the grid;
  3. multiply by the lifted FD operator of the action;
  4. accumulate into the (σ_out, σ_in) sub-block;
  5. assemble the nine sub-blocks into one (3N, 3N) CSR matrix in spin
     order (-1, 0, +1) for rows and columns.

`assemble_vector_spin_block` is that loop. The only caller-specific
piece — how a term coefficient becomes a grid diagonal — is injected as
the `coef_to_diag` callable.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import scipy.sparse as sps
import sympy as sp

from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.term_classify import (
    _decompose_v_expr,
    _action_op,
    _stencils_for_div,
)

SPIN_ORDER = (-1, 0, +1)

# Maps a SymPy coefficient expression to a sparse diagonal on the flat
# (N_r·N_theta) grid. Builders whose coefficients depend on (r, θ) only
# pass `term_classify._coef_to_diag`; advection and Dot(bg_vec, tensor)
# pass a closure that also binds the background's spin components.
CoefToDiag = Callable[[sp.Expr], sps.spmatrix]


def assemble_vector_spin_block(
    expr_per_sigma: Dict[int, sp.Expr],
    v_placeholder_to_id: Dict[sp.Function, Tuple[str, int]],
    bundle: DerivativeBundle,
    m: int,
    coef_to_diag: CoefToDiag,
    *,
    v_unknown_name: str,
    builder_name: str,
    allowed_actions: "frozenset[str] | None" = None,
    origin_p_op: "int | None" = None,
    origin_bg_values: "Dict | None" = None,
) -> sps.csr_matrix:
    """Assemble a (3N, 3N) vector→vector spin block from closed-form
    per-output-spin SymPy expressions.

    Parameters
    ----------
    expr_per_sigma : dict {sigma_out: sp.Expr}
        One expression per output spin, linear in the v-placeholder
        functions and their r/θ derivatives up to second order.
    v_placeholder_to_id : dict {sp.Function: (unknown_name, sigma_in)}
        Maps each v spin-component placeholder to its (unknown, σ_in).
    bundle : DerivativeBundle
        Provides the FD stencils (via `_stencils_for_div`).
    m : int
        Azimuthal mode (passed to the stencil cache).
    coef_to_diag : callable (sp.Expr) -> sparse diagonal
        Turns a term coefficient into a (block, block) diagonal.
    v_unknown_name : str
        Name of the unknown; every term must reference it.
    builder_name : str
        Caller name for error messages.
    allowed_actions : frozenset of str, optional
        If given, every term's action must be in this set; otherwise a
        NotImplementedError is raised. None allows every action that
        `_action_op` supports.
    origin_p_op : int, optional
        The r-power of the expressions (r^p·op). When given, on a ball
        grid the i_r=0 rows are replaced by the operator's r→0 limit
        (`ball_origin.install_origin_rows`), with `origin_bg_values`
        ({applied background function: grid values}) expanded too.

    Returns
    -------
    sps.csr_matrix
        (3N, 3N) block in spin order (-1, 0, +1), N = N_r · N_theta.
    """
    grid = bundle.grid
    block = grid.N_r * grid.N_theta
    stencils = _stencils_for_div(bundle, m)
    sub_blocks: Dict[Tuple[int, int], sps.csr_matrix] = {
        (s_o, s_i): sps.csr_matrix((block, block), dtype=complex)
        for s_o in SPIN_ORDER for s_i in SPIN_ORDER
    }
    for sigma_out, expr in expr_per_sigma.items():
        for term in _decompose_v_expr(expr, v_placeholder_to_id):
            if term["unknown_name"] != v_unknown_name:
                raise RuntimeError(
                    f"{builder_name}: term references unknown "
                    f"{term['unknown_name']!r} != {v_unknown_name!r}"
                )
            action = term["action"]
            if allowed_actions is not None and action not in allowed_actions:
                raise NotImplementedError(
                    f"{builder_name}: unexpected term action {action!r}; "
                    f"allowed actions are {sorted(allowed_actions)}. Term: {term}"
                )
            sigma_in = term["sigma_in"]
            coef_diag = coef_to_diag(term["coef"])
            op = _action_op(action, stencils, block)
            contrib = (coef_diag @ op).tocsr()
            sub_blocks[(sigma_out, sigma_in)] = (
                sub_blocks[(sigma_out, sigma_in)] + contrib
            ).tocsr()
    block = sps.bmat([
        [sub_blocks[(-1, -1)], sub_blocks[(-1,  0)], sub_blocks[(-1, +1)]],
        [sub_blocks[( 0, -1)], sub_blocks[( 0,  0)], sub_blocks[( 0, +1)]],
        [sub_blocks[(+1, -1)], sub_blocks[(+1,  0)], sub_blocks[(+1, +1)]],
    ], format="csr", dtype=complex)
    if origin_p_op is not None:
        from spdee.backends.fd2.ball_origin import install_origin_rows
        block = install_origin_rows(
            block, expr_per_sigma, v_placeholder_to_id, bundle, origin_p_op,
            SPIN_ORDER, SPIN_ORDER, origin_bg_values,
        )
    return block
