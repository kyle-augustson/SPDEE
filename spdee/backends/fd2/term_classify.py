"""spdee.backends.fd2.term_classify — spin-term classification kernel.

Turns a closed-form spin-basis expression, linear in vector-unknown
placeholder functions and their r/θ derivatives up to second order,
into discretized sparse operator pieces. Every vector operator builder
in this backend (advection, Dot(bg_vec, tensor), Curl, Grad(Div), Lap,
Div(tensor)) uses the same three steps:

  1. `_decompose_v_expr` splits a per-output-spin SymPy expression into
     additive term-dicts {coef, sigma_in, action, unknown_name};
  2. `_coef_to_diag` evaluates a term coefficient on the (r, θ) grid as
     a sparse diagonal;
  3. `_stencils_for_div` / `_action_op` map an action tag to its lifted
     FD operator.

`spdee.backends.fd2.spin_block_assembly.assemble_vector_spin_block`
wires these into the shared 3×3 spin-block loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import sympy as sp
import scipy.sparse as sps

from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.operators_core import (
    lift_r_op,
    lift_theta_op,
    diag_from_2d_array,
)
from spdee.ir.fields import r as r_sym, theta as t_sym

# Action tags: how a term acts on the placeholder v^{sigma_in}.
ACTION_POINTWISE = "pointwise"
ACTION_D_R = "d_r"
ACTION_D_T = "d_theta"
ACTION_D_RR = "d_rr"
ACTION_D_TT = "d_theta_theta"
ACTION_D_RT = "d_r_theta"


def _decompose_v_expr(
    expr: sp.Expr,
    v_placeholder_to_id: Dict,
) -> List[Dict[str, Any]]:
    """Decompose a SymPy expression, linear in the v placeholders and
    their derivatives (up to second order, mixed ∂_r∂_θ included), into
    additive term-dicts

        [{"coef": sp.Expr, "sigma_in": int, "action": str,
          "unknown_name": str}, ...]

    where action is one of ACTION_* and coef depends on (r, θ) only.
    """
    expanded = sp.expand(expr)
    if expanded == 0:
        return []
    term_args = expanded.args if isinstance(expanded, sp.Add) else (expanded,)
    out = []
    for term in term_args:
        if term == 0:
            continue
        info = _classify_v_term(term, v_placeholder_to_id)
        if info is None:
            raise ValueError(
                f"term_classify._decompose_v_expr: term {term} contains "
                f"no v placeholder; expression should be linear in v."
            )
        out.append(info)
    return out


def _classify_v_term(
    term: sp.Expr,
    v_placeholder_to_id: Dict,
) -> Dict[str, Any] | None:
    """Classify one additive term as (coefficient) · (action on v_σ)."""
    appearances = _find_v_placeholder_atoms(term, v_placeholder_to_id)
    if not appearances:
        return None
    if len(appearances) > 1:
        raise ValueError(
            f"Term {term} contains multiple v placeholder appearances; "
            f"expression should be linear in v."
        )
    sub_expr, action, applied = appearances[0]
    unknown_name, sigma = v_placeholder_to_id[applied.func]
    # The coefficient is one product term divided by one of its factors,
    # so rational cancellation is all that is needed.
    coef = sp.cancel(term / sub_expr)
    for fn in coef.atoms(sp.Function):
        if hasattr(fn, "func") and fn.func in v_placeholder_to_id:
            raise ValueError(
                f"Term coefficient {coef} still contains a placeholder "
                f"after extraction — term was not properly linear."
            )
    return {
        "coef": coef,
        "sigma_in": sigma,
        "action": action,
        "unknown_name": unknown_name,
    }


def _find_v_placeholder_atoms(
    expr: sp.Expr,
    v_placeholder_to_id: Dict,
) -> List[Tuple[sp.Expr, str, sp.Expr]]:
    """Find all outermost v-placeholder appearances in `expr`: bare
    AppliedUndef or AppliedUndef wrapped in a Derivative of order ≤ 2
    in r and θ. Returns (matched node, action, applied placeholder).
    """
    results = []

    def visit(node):
        if (
            hasattr(node, "func")
            and node.func in v_placeholder_to_id
            and not isinstance(node, sp.Derivative)
        ):
            results.append((node, ACTION_POINTWISE, node))
            return
        if isinstance(node, sp.Derivative):
            inner = node.expr
            if (
                hasattr(inner, "func")
                and inner.func in v_placeholder_to_id
            ):
                vars_ = list(node.variables)
                count_r = sum(1 for v in vars_ if v == r_sym)
                count_t = sum(1 for v in vars_ if v == t_sym)
                total = count_r + count_t
                if total != len(vars_):
                    raise NotImplementedError(
                        f"Derivative {node} contains variables other than "
                        f"r, theta."
                    )
                if total > 2:
                    raise NotImplementedError(
                        f"Derivative {node} is higher than second order; "
                        f"only up to 2nd-order derivatives are supported."
                    )
                if total == 1:
                    action = ACTION_D_R if count_r == 1 else ACTION_D_T
                elif total == 2:
                    if count_r == 2:
                        action = ACTION_D_RR
                    elif count_t == 2:
                        action = ACTION_D_TT
                    else:
                        action = ACTION_D_RT
                else:
                    action = ACTION_POINTWISE
                results.append((node, action, inner))
                return
        for arg in getattr(node, "args", ()):
            visit(arg)

    visit(expr)
    return results


def _check_finite_coefficient(coef_2d: np.ndarray, grid, label) -> None:
    """Raise when a coefficient evaluated on the grid is not finite. The
    r^p-multiplied spin formulas are polynomial in r and finite on the
    pole-free θ grid, so a non-finite entry means an unreduced 1/r or a
    singular background, never something to zero silently."""
    nonfinite = ~np.isfinite(coef_2d)
    if np.any(nonfinite):
        i, j = np.argwhere(nonfinite)[0]
        raise ValueError(
            f"Operator coefficient {label} is not finite at "
            f"{int(nonfinite.sum())} grid node(s), e.g. (r, θ) = "
            f"({grid.r[i]:.6g}, {grid.theta[j]:.6g})."
        )


def _coef_to_diag(coef_expr: sp.Expr, grid) -> sps.spmatrix:
    """Evaluate a SymPy coefficient in (r, θ) on the grid and return a
    sparse diagonal of shape (N_r·N_theta, N_r·N_theta). A non-finite
    entry raises (`_check_finite_coefficient`).
    """
    N_r, N_theta = grid.N_r, grid.N_theta
    coef_fn = sp.lambdify((r_sym, t_sym), coef_expr, modules="numpy")
    R = grid.r[:, None] * np.ones((1, N_theta))
    T = np.ones((N_r, 1)) * grid.theta[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coef_2d = coef_fn(R, T)
    coef_2d = np.broadcast_to(
        np.asarray(coef_2d, dtype=complex), (N_r, N_theta)
    ).copy()
    _check_finite_coefficient(coef_2d, grid, coef_expr)
    return diag_from_2d_array(coef_2d)


def _stencils_for_div(bundle: DerivativeBundle, m: int):
    """Lifted 2D first- and second-derivative stencils in r and θ, at
    the bundle's `order_r` / `order_theta`, cached on the bundle.
    """
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    from spdee.backends.fd2.stencils import standard_d1, standard_d2
    if not hasattr(bundle, "_div_sparse_stencils"):
        _, B_1_r_std = standard_d1(grid.r, order=bundle.order_r)
        _, B_2_r_std = standard_d2(grid.r, order=bundle.order_r)
        _, B_1_t_std = standard_d1(grid.theta, order=bundle.order_theta)
        _, B_2_t_std = standard_d2(grid.theta, order=bundle.order_theta)
        bundle._div_sparse_stencils = {
            "B_1_r_lift": lift_r_op(B_1_r_std, N_theta),
            "B_2_r_lift": lift_r_op(B_2_r_std, N_theta),
            "B_1_t_lift": lift_theta_op(B_1_t_std, N_r),
            "B_2_t_lift": lift_theta_op(B_2_t_std, N_r),
        }
    return bundle._div_sparse_stencils


def _action_op(action: str, stencils: Dict[str, sps.spmatrix], block: int):
    """The (block, block) sparse operator implementing one action tag."""
    if action == ACTION_POINTWISE:
        return sps.eye(block, dtype=complex, format="csr")
    if action == ACTION_D_R:
        return stencils["B_1_r_lift"]
    if action == ACTION_D_T:
        return stencils["B_1_t_lift"]
    if action == ACTION_D_RR:
        return stencils["B_2_r_lift"]
    if action == ACTION_D_TT:
        return stencils["B_2_t_lift"]
    if action == ACTION_D_RT:
        return stencils["B_1_r_lift"] @ stencils["B_1_t_lift"]
    raise ValueError(f"Unknown action: {action!r}")
