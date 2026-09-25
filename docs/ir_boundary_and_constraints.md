# IR contracts for capability parity: boundary rows and constraints

Two IR records carry what the four numerical cores (collocated FD, staggered
Arakawa-C, sh_fd spin-weighted harmonics, Dedalus) need in order to treat the
same system the same way: the composite boundary row `BC_LINEAR_COMBINATION`
(`spdee/ir/tasks.py`, emitted by `spdee/ir/walker.py`) and the constraint
classification `ClassificationRecord` (`spdee/ir/constraints.py`). This page
states the contracts and the parity matrix; the collocated column records
what that core does after consuming them.

## 1. Composite boundary rows

### The record

`walk_boundary_conditions(system, m, composite=True)` returns one `BCTask` per
`BoundaryCondition`. A row with one unknown term keeps its single-shape kind
(`BC_DIRICHLET_SCALAR`, `BC_DIRICHLET_VECTOR_PROJECTION`,
`BC_SCALAR_LINEAR_FROM_TENSOR`, `BC_NEUMANN_SCALAR_RADIAL`,
`BC_PROJECTION_CURL_CROSS`), byte for byte as before; a coefficient on that
single term is divided into the value (`2 p = 3` is `p = 3/2`). A row with
several unknown terms, a Sum / Negation / ScalarMultiple tree over those five
shapes with background coefficients, is

    kind          BC_LINEAR_COMBINATION
    location      'r=0' | 'r=1'
    unknown_name  the host unknown (metadata['host_unknown'])
    value         BackgroundComponent (SymPy) or None: sum_i c_i X_i = value
    metadata
      'terms'        [(c_i, term_i)]: c_i a BackgroundComponent (number,
                     SymPy expression in r, theta and the parameters, or a
                     numeric background; the sign of the term is in c_i),
                     term_i a homogeneous BCTask of one of the five kinds at
                     the same location
      'unknowns'     every unknown the row touches, first-seen order
      'host_unknown' the highest-rank unknown touched (first seen on a tie)
      'host_sigma'   +1 (theta_hat row), -1 (phi_hat row), 0 (other) on a
                     vector host; None on a scalar host
      'tangent'      'theta_hat' | 'phi_hat' | 'mixed' | None
      'pairing_key'  (location, expression structure with theta_hat and
                     phi_hat made alike), `boundary_pairing_key(bc)`

`Dot(u_hat, X)` with X a tree distributes the unit over X, and its Curl /
Cross leaves are gathered back into one curl-cross term, so
`dot(theta_hat, v + ls*dot(strain, r_hat))` and
`dot(theta_hat, v) + ls*dot(theta_hat, dot(strain, r_hat))` give the same
terms. A background tensor inside a projection moves to the value scaled by
the term's coefficient. A term that is none of the five shapes is refused
naming the term; a row whose value is not SymPy-backed is refused.
`bind_bc_parameters(bc_tasks, params)` binds symbolic parameters in values,
coefficients and nested terms.

Without `composite` (the default until every core consumes the record) the
walker refuses composite rows exactly as before ("both sides of the Sum
contain unknowns"), so the cores that read the single-shape kinds see no
change.

### What a core must do

The row at the wall r = r_loc is `sum_i c_i(r_loc, theta) row_i = value`,
where `row_i` is the row the core would assemble for `term_i` alone: the wall
value of a scalar, the unit projection of a vector (`r_hat . v = v^0`,
`theta_hat . v = (v^+ + v^-)/sqrt2`, `phi_hat . v = i(v^- - v^+)/sqrt2`), the
one-sided radial derivative of a scalar, the `scalar_expr` of a tensor
projection with its r and theta derivatives, the curl / cross combination
projected on its unit. The coefficient is evaluated at the wall: an FD core
at every wall node (theta dependence allowed), a spectral core as a number
or a radial profile (a theta-dependent coefficient couples the harmonics of
the wall and is refused). The row replaces (FD) or is lifted into (tau) the
rows of the host: on a vector host the spin block `host_sigma` at the wall.
A row naming two unknowns (the normal stress on v and p) is one row on the
host with columns of both.

Pairing rule (the Dedalus rule, `_boundary_rows`): a theta_hat row and a
phi_hat row on the same expression at the same wall, the same
`pairing_key`, are the two components of one tangential vector condition.
A spectral core with vector harmonics (Dedalus, sh_fd) imposes them as one
angular row and refuses a lone tangential row; an FD core imposes them as
two scalar rows, theta_hat on sigma = +1 and phi_hat on sigma = -1 of the
host. A 'mixed' row (theta_hat and phi_hat in one expression) is a scalar
row on an FD core and is refused by the spectral cores. The single-shape
kinds pair the same way through `boundary_pairing_key(bc)` on the system's
`BoundaryCondition`, which the tasks are listed in order of.

## 2. Constraint classification

### The record

`classify(system, tasks, *, geometry, mimetic, core="constraints")` returns a
`ClassificationRecord` (lifted from
`staggered_general.classify_constraints`, whose dict it reproduces in
`staggered_view()` field by field, refusals and messages included;
`tests/test_constraints.py::test_agrees_with_the_staggered_classifier`).

    geometry, mimetic, m
    dt_less              the system has no Dt: it is the eigenvalue operator
                         itself, and a vector equation whose only vector
                         unknown is u is u's evolution with a unit mass
                         ('implicit_mass' on the equation)
    rank                 {unknown: 0 | 1 | 2}
    placement            {unknown: 'center' | 'face' | 'tensor' | 'tensor9'}
    equations            {equation: {'class': 'evolution' | 'saddle_point' |
                         'involution' | 'definition', ...}}
    principal            {equation: unknown whose DOFs carry its rows}
    multipliers          {multiplier: constraint equation}
    involutions          {vector: its curl evolution equation}
    constraints          {vector: 'saddle_point' | 'involution'}
    definitions          {tensor: defining equation}
    gauges               {multiplier: 'integral' | 'point' | None}
    involution_treatment 'dropped' (mimetic) | 'saddle_point' (non-mimetic)
    divergence_cleaning  {vector: DivergenceCleaning}, non-mimetic only

Classes and their payload:

* evolution: `unknown`; `implicit_mass` under the Dt-less rule.
* saddle_point: `Div(M u) = 0` with a scalar p entering u's evolution through
  Grad alone: `vector`, `multiplier`, `weight` (M or None), `div_sign`,
  `div_outer`, `grad_tasks`, `gauge` ('integral' when p is a declared gauge
  unknown, 'point' otherwise, at m = 0; None at m > 0). The multiplier's rows
  are the constraint's rows with zero mass.
* involution: `Div(b) = 0` with no multiplier and a curl evolution (plus
  `Lap(b)` or `Curl(Curl(b))`): `vector`, `evolution`, `dropped`. Implied
  involutions (no written constraint) appear in `involutions` and
  `constraints` only.
* definition: `T - tensor(u) = 0` without Dt: `tensor`, `identity`,
  `terms` [(task, (a, b))] with constant a Grad(u) + b Transpose(Grad(u))
  weights, `vector`; placement 'tensor' when every term is symmetric, else
  'tensor9'.
* refusals (NotImplementedError, the C-grid messages with the `core`
  prefix): an algebraic relation without a multiplier, a divergence with
  neither Grad nor Curl, a tensor unknown with no tensor equation, both
  readings (a multiplier and a curl evolution), a weighted involution, a
  non-curl term in an involution's evolution, several candidate multipliers,
  Dt on several unknowns, an unknown under Dt twice, a leftover unknown, a
  definition mixing vectors, a taken `__div_<b>` name.

### What a core must do

* mimetic (C-grid): as today. A written involution row is dropped, b keeps
  the face fluxes, `Lap(b)` is `-Curl Curl b`; the wall demotion of a
  self-advected vector (`_demote_pinned_involutions`) stays a wall-record
  step after classification.
* non-mimetic (collocated, sh_fd, Dedalus): `involution_treatment` is
  'saddle_point' and every involution b carries a `DivergenceCleaning`:
  introduce the scalar `multiplier` (`__div_b`, rank 0, 'center'), add
  `+ Grad(multiplier)` to `evolution`, host the row `Div(b) = 0` on the
  multiplier's DOFs (`constraint` names the written equation; None means the
  core adds the row), pin the multiplier to zero at every wall in `walls`
  (the outer wall on a ball) and impose the mean-zero `gauge` at m = 0. This
  is the divergence-cleaning projection: the multiplier is the Lagrange
  multiplier of `Div(b) = 0` and vanishes on the exact solution.
* both: `placement` / `rank` fix the DOF blocks; `dt_less` selects the unit
  mass on the promoted vector equation; `gauges` names the m = 0 gauge of
  every multiplier.

## 3. Parity matrix (what each core does today)

Read from the consumers (`fd2/compiled_problem.py`, `fd2/staggered_general.py`,
`sh_fd/compiler.py`, `dedalus/compiler.py`) and checked by compiling each
row on a shell at m = 1; "verified" means an analytic benchmark exists.

| row | collocated | staggered | sh_fd | Dedalus |
|---|---|---|---|---|
| Robin / Biot `dot(r_hat, grad(p)) + h p = 0` | verified on ball and shell (`test_collocated_parity.py::test_helmholtz_robin_*`, `test_biot_wall_binds_the_parameters`): the wall row is the per-node sum of the coefficient times the term rows, on the host of the record | verified on ball and shell (`test_ball_staggered.py::TestCompositeWallsBall::test_robin_decay_rates_second_order`, `test_staggered_general.py::TestCompositeWallsShell::test_robin_decay_rates_second_order`): the mirror-ghost flux closure with the Robin coefficient per wall node, wall flux g s_0 with g = -c_D / (c_N + sigma c_D h/2), added to the natural stencils (`_robin_wall_term`); k j_l' + h j_l = 0 and the two-wall determinant at order 2.0; h = 0 the Neumann pencil, c_N = 0 the Dirichlet closure; order 2 only (refused at order > 2) | verified on ball and shell (`test_sh_fd_parity.py::test_helmholtz_robin`, `test_thermal_diffusion_radiative_wall`, every coefficient spelling): `_lin(bc.expr)` translates the row directly on p's wall node, order 3.7-4 against k j_l' + h j_l = 0 and the shell determinant | verified on ball and shell (`test_dedalus_ball.py::test_helmholtz_robin`, `test_dedalus_parity.py::test_helmholtz_robin_shell`, `test_thermal_diffusion_biot_shell`, every coefficient spelling on both): the row is the sum of the term rows at the wall, the coefficient a number there |
| Navier slip `dot(t_hat, v) + ls dot(t_hat, dot(strain, r_hat)) = 0` (theta / phi pair) | verified on the ball (`test_navier_slip_ball`): two scalar rows on sigma = +1 and -1 of v | verified on the ball (`TestCompositeWallsBall::test_navier_slip_toroidal_rates_second_order`, shell limits in `TestCompositeWallsShell`): the strain closure with the slip length per wall node (`TangentialWall`), the ghost g_g = gamma g_0 of v_t + ls (d_r v_t - v_t/r) = 0 on the divided field (shell) or v_g = rho v_0 (ball), the wall shear faces at weight 1/2 + ls/h; (1 - ls) j_l + ls k j_l' = 0 at order 2.0; ls = 0 and ls = inf are the no-slip and stress-free pencils bitwise, one code path; no-slip inside with stress-free outside is ls = (0, inf) | verified on ball and shell (`test_vector_diffusion_navier_slip`, `test_navier_slip_limits`): the pair on the record's host_sigma blocks (+1 theta_hat, -1 phi_hat) imposes the two spin components of one tangential condition exactly; (1 - ls) j_l + ls k j_l' = 0 at order 3.7, ls -> 0 / oo the no-slip / stress-free roots | verified on ball and shell (`test_vector_diffusion_navier_slip`, `test_vector_diffusion_navier_slip_shell`): one angular row from the pair, resolved term by term on both geometries (`_wall`) |
| normal stress `dot(r_hat, dot(strain, r_hat)) - c p = 0` | verified on the ball (`test_normal_stress_ball`): one row on sigma = 0 of v with columns of v and p, linear in c | verified on the ball (`TestCompositeWallsBall::test_normal_stress_bessel_determinant`, c = 1, 2; shell structure in `TestCompositeWallsShell`): an open wall, the wall v_r faces stay DOFs with the half-cell rows of the strain Gram (natural traction) and the Dirichlet-closed gradient (the wall cell's pressure), plus the residual traction -sigma (2/h)(c_l kappa - c_g) p_w, p_w extrapolated at second order; the m = 0 gauge lifted; the traction-free determinant roots and the toroidal family at order 1.9-2.0 | verified on the ball (`test_normal_stress_row`): one row on sigma = 0 of v (the record's host) with the columns of v and p, linear in the IR, toroidal and traction-free roots at order 3.5-4; the l = 0 tau mode 10/c of the Dedalus ball has no row-replacement counterpart | verified on ball and shell (`test_normal_stress_row`, `test_normal_stress_row_shell`): row linear in the IR, constrains p, tau layout of the stress-free wall |
| tensor unknowns (`dt(T) + T - strain = 0`) | supported: nine covariant spin blocks (`tensor_unknown_sparse.py`), rank-2 origin regularity on the ball (`tensor_origin_basis`), the tensor kinds dispatched (`test_tensor_under_dt_keeps_nine_blocks`, `test_tensor_blocks_on_manufactured_fields`) | supported: six symmetric or nine components on the strain locations (section 2b) | supported: nine spin blocks T^{ab} with the rank-2 ghost closure on the ball and the absent spin pm2 rows pinned (`test_tensor_under_dt_keeps_its_blocks`, `test_sh_fd_ball.py::test_rank2_regularity_at_the_origin`); no wall rows on a tensor | supported: T stays in the state (the first-order system is native) with no tau of its own, the IR having no radial derivative of a rank-2 unknown; rows written on T are lifted on its vector (`_tau_owner`); the relaxed strain splits every toroidal rate Ek k^2 into the roots of lambda^2 - lambda + Ek k^2 (`test_relaxed_tensor_splits_every_decay_rate`) |
| definitions (`T - strain = 0`, T eliminated) | supported: classified 'definition', the tasks on T rewritten into the strain-form tasks on u, entrywise the strain pencil (`test_tensor_definition_is_eliminated`) | supported: classified 'definition', composed with the definition map | supported: classified 'definition', T = -rest/c substituted in `_lin`, entrywise the strain pencil (`test_definition_is_eliminated`) | supported: T kept as a variable, the strain-form spectrum to rounding on the ball and to the tau-polynomial difference on the shell (`test_definition_keeps_the_tensor`) |
| involution as saddle point (`Div(b) = 0` written, no multiplier) | verified on shell and ball (`test_free_decay_b_formulation`): the `DivergenceCleaning` multiplier `__div_b`, its Grad in the induction equation, Dirichlet-zero walls, mean-zero gauge at m = 0; with the user's psi an ordinary saddle point; implied involution: no constraint (unchanged) | supported: the row is dropped, CT fluxes | verified on shell and ball (`test_magnetic_free_decay_b`): the `DivergenceCleaning` multiplier `__div_b`, its Grad in the induction equation, Dirichlet-zero walls on the divergence rows at the wall nodes; the m = 0 mean-zero gauge is not imposed on top of the walls (it admits a spurious l = 0 mode); implied involution: no constraint (unchanged) | verified on shell and ball (`test_written_involution_is_cleaned`): the `DivergenceCleaning` multiplier `__div_b` hosted as a gauge scalar, + Grad in the induction equation, `integ` gauge (l = 0 only); the wall pins are not taken (a tau core has no wall DOF on the multiplier, which vanishes on the eigenvectors at rounding); the pencil is the hand-written psi's entry for entry; with the user's psi as written; implied: no constraint |
| vector potential (`dt(A) - cross(v, B0) - eta lap(A) - grad(psi) = 0`, `div(A) = 0`) | verified (`test_free_decay_vector_potential`, `test_dipole_rossby_shift_b_and_A_formulations`): `cross_curl_curl_unk_bg` is the Cross block times the r^2 Curl(Curl) block | supported (`ct_A`): saddle point with tangential A = 0 and psi = 0 walls | verified on shell and ball (`test_vector_potential_free_decay`, `test_dipole_shift_A_matches_b`): psi's rows are the Coulomb rows and its walls replace them at the wall nodes (the tau owner's slot on A loses an order, docs/ball_sh_fd.md 6); the dipole shift agrees with the b formulation to 2.5% at 16 x 16 | supported: psi's taus handed to A (`_tau_owner`) |
| LapH | supported (`TASK_LAPH_SCALAR`) | supported; at order > 2 a Dirichlet wall reflects the flux stencils with the odd parity (`_stag_diff_1d_wall_parity`), so the transposed divergence stays consistent through the wall cells: the wall cells converge at order 2 (the mirror ghost's order, as the natural closure and the pole / origin rows do) while the interior keeps the stencil order; the coverage cells at order_r = 4 measure p = 2.0-2.1 on the shell with errors 2-4x below order 2 | supported (`-l(l+1)/r^2`) | supported: `lap - (2 x.grad + x.grad(grad).x) / r^2` with x = r r_hat, 1/r^2 an NCC on the shell and probed at the nodes on the ball (`test_laph_identity`; coverage cells pass at m = 0, 1, 2) |

What the phase-two cores fill in, from the records:

* collocated (done, docs/ball_collocated.md): `BC_LINEAR_COMBINATION`
  consumed by summing the term rows (Dirichlet, Neumann, projection, tensor
  projection, curl-cross) on the host row; `DivergenceCleaning` taken for a
  written involution (an implied one keeps today's constraint-free pencil);
  `cross_curl_curl_unk_bg` built as the Cross block times the Curl(Curl)
  block; tensor unknowns placed on nine covariant spin blocks with the
  rank-2 origin regularity, definitions eliminated by task rewriting.
* staggered (done, docs/staggered_general_design.md 5): `classify_constraints`
  is `classify(mimetic=True).staggered_view()`; composite rows consumed as
  the C-grid closures, Robin on a centre scalar as the mirror-ghost flux
  closure with the coefficient, Navier slip as the strain closure with a
  slip length per wall node (the no-slip and stress-free closures its
  limits, bitwise), the normal stress as an open wall whose v_r faces
  stay DOFs; any other composite shape refused naming the term.
* sh_fd (done, docs/ball_sh_fd.md 5a and 6): composite rows hosted on the
  record's host (single-shape rows keep the strongest-block rule, whose
  theta / phi tie is a rounding-level row permutation that the certified
  pencils fix); a theta / phi pair is exact, a lone or mixed tangential row
  is the Galerkin projection on the spin-0 harmonics rather than refused;
  tensor unknowns on nine spin blocks, definitions eliminated;
  `DivergenceCleaning` for a written involution with the walls on the
  divergence rows and without the m = 0 gauge row; psi of the vector
  potential on its constraint rows rather than the tau owner's slot; the
  mean-zero gauge row of a declared gauge unknown at m = 0.
* Dedalus (done, tests/test_dedalus_parity.py): the shell rows resolved term
  by term through `_wall` as on the ball (a tangential unit vector is not an
  NCC of the sphere), the pairing generalised to the 2^k combinations of k
  nested tangential projections; `DivergenceCleaning` taken for a written
  involution (an implied one keeps the constraint-free pencil, Div Curl = 0
  holding on the polynomial space); tensor unknowns kept in the state with
  their rows handed to their vector; LapH as lap minus its radial part.

## Fixing the multiplier's constant

The multiplier of a divergence cleaning has a null space that must be fixed
exactly once.  A core that can impose rows on the multiplier takes the
Dirichlet walls of the record; a tau core, which has no wall-row layout for
the multiplier, takes the mean-zero gauge alone.  Imposing both admits one
spurious mode of the vector (sh_fd measured it at |lambda| ~ 8800 on the
shell; the collocated core at 4700 on the shell and 1800 on the ball), so the
walls and the gauge are alternatives.  The collocated core is the exception
on a ball at m = 0: its gradient has a checkerboard null vector in the
multiplier there, zero at the origin ring and at the wall, which the walls
cannot see, so it imposes the mean-zero row as well and carries that one
spurious mode far from the physical band; on the shell it takes the walls
alone and is clean.
