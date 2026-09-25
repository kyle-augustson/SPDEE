"""spdee.backends.fd2 — finite-difference backend on spherical (r, θ) grids.

Explicit centered finite-difference stencils of even order, chosen
independently for the radial and angular axes through the `order_r`
and `order_theta` arguments of `compile_problem` / `compile_fd2`
(default 4 for both). Any even order ≥ 2 is accepted; higher orders
widen the bands and converge faster on smooth problems.

Modules:
  - `grid`: `Grid`, `make_ball_grid`, `make_shell_grid`.
  - `stencils`: 1D first- and second-derivative matrices (uniform,
    non-uniform, and parity-folded variants for r=0).
  - `operators`: `DerivativeBundle` (stencil cache) and the equation
    mass matrices `r1_*`, `r2_*`, `r_sqrt2_vector_mass`.
  - `lap_sparse`, `curl_cross_sparse`, `grad_div_sparse`,
    `div_tensor_sparse`, `advection_sparse`,
    `dot_bg_vec_tensor_unk_sparse`: sparse builders for each
    differential operator, driven by `spin_block_assembly` and
    `term_classify`.
  - `compiler`: walks an `EquationSystem` and assembles A_total,
    B_total, b_total.
  - `compiled_problem`: applies origin regularity, gauge constraints
    and boundary conditions; exposes `solve_bvp` / `solve_evp`.
  - `evp_filter`: twin-grid spurious-mode detection for EVPs.

Public entry point: `spdee.backends.fd2.compiled_problem.compile_problem`.
"""


def compile(system, **kwargs):
    """Backend-registry entry point; forwards to `compile_problem`."""
    from spdee.backends.fd2.compiled_problem import compile_problem
    return compile_problem(system, **kwargs)
