"""spdee.backends.fd2.tensor_unknown_sparse — blocks of a rank-2 unknown.

A tensor unknown T is stored in its nine covariant spin components
T_{σ_a σ_b} (layout.SPIN_COMPONENTS_TENSOR order), each an (N_r, N_θ)
block like a scalar; the components are the ones the reducer produces
for a tensor expression, so a definition T - tensor(u) = 0 is
componentwise. Every builder here follows the spin-block recipe of
`spin_block_assembly` with the row and column spin sets generalised to
index pairs:

  - `build_r_tensor_of_vector_block`: the (9N, 3N) block of r T_{σ_a σ_b}
    in a vector unknown's spin components (TASK_TENSOR_OF_VECTOR, the
    walker's ``rT_components``), r-power 1;
  - `build_r2_div_tensor_unk_block`: the (3N, 9N) block of r² (Div T)^σ
    on the components of T (TASK_DIV_TENSOR_UNK), r-power 2, from the
    U = r T form of `r2_div_tensor_spin_components`;
  - `build_dot_bg_vec_unk_tensor_block`: the pointwise (3N, 9N) block of
    Dot(a, T) or Dot(T, a) with a background vector a
    (TASK_DOT_BG_VEC_UNK_TENSOR), r-power 0;
  - `r1_tensor_mass`: diag(r) on the nine blocks with the ball's identity
    placeholders at i_r = 0.

On a ball the i_r = 0 rows of the differential blocks are the
operator's r → 0 limit (`ball_origin.install_origin_rows`).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sps
import sympy as sp

from spdee.backends.fd2.layout import SPIN_COMPONENTS_TENSOR, SPIN_COMPONENTS_VECTOR
from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.operators_core import diag_from_2d_array
from spdee.backends.fd2.term_classify import (
    _action_op, _coef_to_diag, _decompose_v_expr, _stencils_for_div,
)
from spdee.ir.fields import r as r_sym, theta as t_sym
from spdee.ir.tensor_reduce import r2_div_tensor_spin_components


def tensor_unknown_placeholders(name: str) -> Dict[Tuple[int, int], sp.Function]:
    """SymPy placeholders {(σ_a, σ_b): T_ab(r, θ)} of a tensor unknown's
    covariant spin components, named ``{name}_sigma_{m|0|p}{m|0|p}``."""
    label = {-1: "m", 0: "0", +1: "p"}
    return {
        key: sp.Function(f"{name}_sigma_{label[key[0]]}{label[key[1]]}")(r_sym, t_sym)
        for key in SPIN_COMPONENTS_TENSOR
    }


def assemble_spin_block(
    expr_per_out: Dict, placeholder_to_id: Dict, bundle: DerivativeBundle, m: int,
    outs, ins, *, unknown_name: str, builder_name: str, origin_p_op=None,
) -> sps.csr_matrix:
    """The (len(outs)·N, len(ins)·N) block of the closed-form expressions
    `expr_per_out` (one per output index in `outs`, linear in the
    placeholders of `placeholder_to_id`, {func: (unknown, input index)})
    with the blocks in the orders `outs` and `ins`; the ball's origin
    rows replace i_r = 0 when `origin_p_op` is given."""
    grid = bundle.grid
    block = grid.N_r * grid.N_theta
    stencils = _stencils_for_div(bundle, m)
    sub = {(o, i): sps.csr_matrix((block, block), dtype=complex) for o in outs for i in ins}
    for out_key, expr in expr_per_out.items():
        for term in _decompose_v_expr(expr, placeholder_to_id):
            if term["unknown_name"] != unknown_name:
                raise RuntimeError(
                    f"{builder_name}: term references unknown "
                    f"{term['unknown_name']!r} != {unknown_name!r}"
                )
            op = _action_op(term["action"], stencils, block)
            contrib = (_coef_to_diag(term["coef"], grid) @ op).tocsr()
            key = (out_key, term["sigma_in"])
            sub[key] = (sub[key] + contrib).tocsr()
    block_mat = sps.bmat([[sub[(o, i)] for i in ins] for o in outs],
                         format="csr", dtype=complex)
    if origin_p_op is not None:
        from spdee.backends.fd2.ball_origin import install_origin_rows
        block_mat = install_origin_rows(
            block_mat, expr_per_out, placeholder_to_id, bundle, origin_p_op, ins, outs,
        )
    return block_mat


def build_r_tensor_of_vector_block(
    rT_components: Dict[Tuple[int, int], sp.Expr],
    v_placeholders: Dict[int, sp.Function],
    bundle: DerivativeBundle,
    m: int,
    v_unknown_name: str,
) -> sps.csr_matrix:
    """(9N, 3N) block with diag(r) · T_{σ_a σ_b} = block · v_state for the
    tensor expression whose r-scaled covariant components are
    `rT_components` in the placeholders of v."""
    placeholder_to_id = {v_placeholders[s].func: (v_unknown_name, s)
                         for s in SPIN_COMPONENTS_VECTOR}
    expr = {key: sp.expand(rT_components[key]) for key in SPIN_COMPONENTS_TENSOR}
    return assemble_spin_block(
        expr, placeholder_to_id, bundle, m, SPIN_COMPONENTS_TENSOR, SPIN_COMPONENTS_VECTOR,
        unknown_name=v_unknown_name, builder_name="build_r_tensor_of_vector_block",
        origin_p_op=1,
    )


def build_r2_div_tensor_unk_block(
    tensor_name: str, bundle: DerivativeBundle, m: int,
) -> sps.csr_matrix:
    """(3N, 9N) block with diag(r²) · (Div T)^σ = block · T_state on the
    nine components of the tensor unknown `tensor_name`."""
    T = tensor_unknown_placeholders(tensor_name)
    placeholder_to_id = {T[key].func: (tensor_name, key) for key in SPIN_COMPONENTS_TENSOR}
    U = {key: r_sym * T[key] for key in SPIN_COMPONENTS_TENSOR}
    div = r2_div_tensor_spin_components(U, m)
    expr = {s: sp.expand(div[s].doit()) for s in SPIN_COMPONENTS_VECTOR}
    return assemble_spin_block(
        expr, placeholder_to_id, bundle, m, SPIN_COMPONENTS_VECTOR, SPIN_COMPONENTS_TENSOR,
        unknown_name=tensor_name, builder_name="build_r2_div_tensor_unk_block",
        origin_p_op=2,
    )


def build_dot_bg_vec_unk_tensor_block(
    a_grid: Dict[int, np.ndarray], bundle: DerivativeBundle, side: str,
) -> sps.csr_matrix:
    """Pointwise (3N, 9N) block of Dot(a, T) (side 'left'),
    (Dot(a, T))^{σ_out} = Σ_{σ_in} a^{σ_in} T_{σ_in, -σ_out}, or Dot(T, a)
    ('right'), Σ_{σ_in} T_{-σ_out, σ_in} a^{σ_in}, with `a_grid` the
    background's contravariant spin components on the grid."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
    grid = bundle.grid
    block = grid.N_r * grid.N_theta
    sub = {}
    for s_out in SPIN_COMPONENTS_VECTOR:
        for key in SPIN_COMPONENTS_TENSOR:
            sub[(s_out, key)] = sps.csr_matrix((block, block), dtype=complex)
        for s_in in SPIN_COMPONENTS_VECTOR:
            key = (s_in, -s_out) if side == "left" else (-s_out, s_in)
            sub[(s_out, key)] = diag_from_2d_array(a_grid[s_in]).tocsr()
    return sps.bmat([[sub[(o, i)] for i in SPIN_COMPONENTS_TENSOR]
                     for o in SPIN_COMPONENTS_VECTOR], format="csr", dtype=complex)


def r1_tensor_mass(bundle: DerivativeBundle) -> sps.csr_matrix:
    """(9N, 9N) mass diag(r) on every component block, with identity
    placeholders on the i_r = 0 rows in ball mode."""
    grid = bundle.grid
    R = grid.r[:, None] * np.ones((1, grid.N_theta))
    if grid.is_ball:
        R = R.copy()
        R[0, :] = 1.0
    one = diag_from_2d_array(R).tocsr()
    return sps.block_diag([one] * len(SPIN_COMPONENTS_TENSOR), format="csr", dtype=complex)
