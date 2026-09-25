"""Sparse builder for the advection operator (v0·∇)v.

Implements the spin-basis contraction
    (v0·∇v)^{σ_3} = Σ_{σ_a} v0^{σ_a} (Grad v)_{σ_a, -σ_3}
with (Grad v)_{σ_a, σ_b} the covariant rank-2 tensor from
`tensor_reduce.grad_vector_spin_components`; the second slot is -σ_3
because the spin metric is g_{σσ'} = δ_{σ,-σ'}.

The angular pieces of Grad carry 1/r, so the block built is
`r · (v0·∇v)^σ` and the equation mass is `diag(r)` per spin block
(`r1_vector_mass`), the same as Cross, Curl and Grad(scalar).

Coupling structure: output σ_3 = -1 couples to v^- and v^0 only,
σ_3 = +1 to v^+ and v^0 only, σ_3 = 0 to all three.

v0 evaluation: v0's spin components enter the formula as undefined
SymPy Functions bm, b0, bp of (r, θ). After expansion and term
decomposition each coefficient contains one of these functions times
factors of (r, θ, m); it is lambdified with the function values as
extra arguments and evaluated with v0's grid arrays, which works the
same for SymPy-backed and tabular (NumericComponent) backgrounds.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.sparse as sps
import sympy as sp

from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.operators_core import diag_from_2d_array
from spdee.backends.fd2.spin_block_assembly import assemble_vector_spin_block
from spdee.backends.fd2.term_classify import _check_finite_coefficient
from spdee.ir.fields import r as r_sym, theta as t_sym
from spdee.ir.tensor_reduce import grad_vector_spin_components


def _r_times_advection_formula(
    v_placeholders: Dict[int, sp.Function],
    b_placeholders: Dict[int, sp.Function],
    m: int,
) -> Dict[int, sp.Expr]:
    """{σ_3: r·(v0·∇v)^{σ_3}} with v0's spin components left as the
    placeholder Functions in `b_placeholders`.
    """
    v_spin_dict = {s: v_placeholders[s] for s in (-1, 0, +1)}
    T_cov = grad_vector_spin_components(v_spin_dict, m)
    result = {}
    for sigma_3 in (-1, 0, +1):
        expr = sum(
            b_placeholders[sa] * T_cov[(sa, -sigma_3)]
            for sa in (-1, 0, +1)
        )
        result[sigma_3] = sp.expand(r_sym * expr)
    return result


def _coef_to_diag_with_b(
    coef_expr: sp.Expr,
    b_placeholder_syms: Dict[int, sp.Symbol],
    b_func_calls: Dict[int, sp.Expr],
    b_grid: Dict[int, np.ndarray],
    grid,
) -> sps.spmatrix:
    """Evaluate a coefficient that may contain the background's
    spin-component placeholder functions on the grid and return a
    sparse diagonal. A non-finite entry (a singular background) raises.

    The function calls are replaced by bare symbols, the expression is
    lambdified in (r, θ, b_-, b_0, b_+), and the values come from
    `b_grid`.
    """
    N_r, N_theta = grid.N_r, grid.N_theta
    expr_sub = coef_expr
    for sa in (-1, 0, +1):
        expr_sub = expr_sub.subs(b_func_calls[sa], b_placeholder_syms[sa])
    fn = sp.lambdify(
        (r_sym, t_sym,
         b_placeholder_syms[-1], b_placeholder_syms[0], b_placeholder_syms[+1]),
        expr_sub, modules="numpy",
    )
    R = grid.r[:, None] * np.ones((1, N_theta))
    T = np.ones((N_r, 1)) * grid.theta[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coef_2d = fn(R, T, b_grid[-1], b_grid[0], b_grid[+1])
    coef_2d = np.broadcast_to(
        np.asarray(coef_2d, dtype=complex), (N_r, N_theta)
    ).copy()
    _check_finite_coefficient(coef_2d, grid, coef_expr)
    return diag_from_2d_array(coef_2d)


def build_r_advection_vector_block(
    b_grid: Dict[int, np.ndarray],
    v_placeholders: Dict[int, sp.Function],
    bundle: DerivativeBundle,
    m: int,
    v_unknown_name: str,
) -> sps.csr_matrix:
    """(3N, 3N) sparse block with `r·(v0·∇v)^σ = block · v_state`.

    Parameters
    ----------
    b_grid : dict {sigma: (N_r, N_theta) ndarray}
        v0's contravariant spin components on the grid.
    v_placeholders : dict {sigma: sp.Function}
        v's spin-component placeholder functions.
    bundle : DerivativeBundle
        Provides the FD stencils.
    m : int
        Azimuthal mode.
    v_unknown_name : str
        Name of the unknown, for error messages.

    Returns
    -------
    sps.csr_matrix
        (3N, 3N) block in spin order (-1, 0, +1), N = N_r · N_theta.
    """
    b_funcs = {
        sa: sp.Function(f"_b_sigma_{sa}") for sa in (-1, 0, +1)
    }
    b_func_calls = {sa: b_funcs[sa](r_sym, t_sym) for sa in (-1, 0, +1)}
    b_placeholder_syms = {
        sa: sp.symbols(f"_b_sigma_{sa}_val") for sa in (-1, 0, +1)
    }
    expr_per_sigma = _r_times_advection_formula(
        v_placeholders, b_func_calls, m,
    )
    v_placeholder_to_id = {
        v_placeholders[s].func: (v_unknown_name, s) for s in (-1, 0, +1)
    }

    def coef_to_diag(coef_expr):
        return _coef_to_diag_with_b(
            coef_expr, b_placeholder_syms, b_func_calls, b_grid, bundle.grid,
        )

    return assemble_vector_spin_block(
        expr_per_sigma, v_placeholder_to_id, bundle, m, coef_to_diag,
        v_unknown_name=v_unknown_name,
        builder_name="build_r_advection_vector_block",
        origin_p_op=1,
        origin_bg_values={b_func_calls[sa]: b_grid[sa] for sa in (-1, 0, +1)},
    )
