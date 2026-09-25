"""Sparse builder for Dot(background_vector, tensor_expression).

Implements the spin-basis contraction
    r·(Dot(a, T))^{σ_out} = Σ_{σ_a} a^{σ_a}(r, θ) · (r·T)_{σ_a, -σ_out}
where T is a tensor expression in an unknown vector v (and possibly
background tensors) and a is a known background vector. The second
covariant slot is -σ_out because the spin metric is
g_{σσ'} = δ_{σ,-σ'}.

The covariant components T_{σ_a, σ_b} arrive from
`reduce_tensor_expression` as `r·T_{σ_a, σ_b}` expressions polynomial
in r (the factor r clears the 1/r of the Christoffel pieces of
Grad(v)), so the result has r-power 1 and pairs with Cross, Curl,
Grad(scalar) and advection under `r1_vector_mass`. A typical use is
the anelastic viscous term ν ∇ln ρ₀ · T.

Evaluation mirrors `advection_sparse`: a's spin components enter the
formula as undefined SymPy Functions, each term's coefficient is
lambdified with their values as extra arguments, and the grid arrays
of a are passed in.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sps
import sympy as sp

from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.advection_sparse import _coef_to_diag_with_b
from spdee.backends.fd2.spin_block_assembly import assemble_vector_spin_block
from spdee.ir.fields import r as r_sym, theta as t_sym


def build_r_dot_bg_vec_tensor_unk_block(
    a_grid: Dict[int, np.ndarray],
    rT_components: Dict[Tuple[int, int], sp.Expr],
    v_placeholders: Dict[int, sp.Function],
    bundle: DerivativeBundle,
    m: int,
    v_unknown_name: str,
) -> sps.csr_matrix:
    """(3N, 3N) sparse block with `r·(Dot(a, T))^σ_out = block · v_state`.

    Parameters
    ----------
    a_grid : dict {sigma: (N_r, N_theta) ndarray}
        a's contravariant spin components on the grid.
    rT_components : dict {(sigma_a, sigma_b): sp.Expr}
        ``r · T_{sigma_a, sigma_b}`` in v-placeholder form, polynomial in r.
    v_placeholders : dict {sigma: sp.Function}
        v's spin-component placeholder functions.
    bundle : DerivativeBundle
        Provides the FD stencils.
    m : int
        Azimuthal mode (already in rT_components; passed to the
        stencil cache).
    v_unknown_name : str
        Name of the unknown, for error messages.

    Returns
    -------
    sps.csr_matrix
        (3N, 3N) block in spin order (-1, 0, +1).
    """
    a_funcs = {
        sa: sp.Function(f"_a_sigma_{sa}") for sa in (-1, 0, +1)
    }
    a_func_calls = {sa: a_funcs[sa](r_sym, t_sym) for sa in (-1, 0, +1)}
    a_placeholder_syms = {
        sa: sp.symbols(f"_a_sigma_{sa}_val") for sa in (-1, 0, +1)
    }
    expr_per_sigma: Dict[int, sp.Expr] = {}
    for sigma_out in (-1, 0, +1):
        acc = sp.S.Zero
        for sigma_a in (-1, 0, +1):
            acc = acc + a_func_calls[sigma_a] * rT_components[(sigma_a, -sigma_out)]
        expr_per_sigma[sigma_out] = sp.expand(acc)
    v_placeholder_to_id = {
        v_placeholders[s].func: (v_unknown_name, s) for s in (-1, 0, +1)
    }

    def coef_to_diag(coef_expr):
        return _coef_to_diag_with_b(
            coef_expr, a_placeholder_syms, a_func_calls, a_grid, bundle.grid,
        )

    return assemble_vector_spin_block(
        expr_per_sigma, v_placeholder_to_id, bundle, m, coef_to_diag,
        v_unknown_name=v_unknown_name,
        builder_name="build_r_dot_bg_vec_tensor_unk_block",
        origin_p_op=1,
        origin_bg_values={a_func_calls[sa]: a_grid[sa] for sa in (-1, 0, +1)},
    )
