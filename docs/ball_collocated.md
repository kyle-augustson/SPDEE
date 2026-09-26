# The origin in the collocated ball

Reference for `spdee/backends/fd2/ball_origin.py`, `compiler.py` and
`CompiledProblem` on `geometry='ball'` grids (`r[0] == 0.0` exactly).
The module docstring of `ball_origin.py` is the normative statement;
this page adds the derivations and the benchmark evidence.

## Regularity at r = 0

A field is Cartesian-smooth. With `f = Σ_l f_lm(r) Y_l^m`, `f_lm ~ r^l`:

| unknown | m | value at r = 0 | DOFs |
|---|---|---|---|
| scalar | 0 | `f(0, θ) = f(0)` | 1 |
| scalar | ≠ 0 | 0 | 0 |
| vector | 0 | `c ẑ`: `v^0 = c cos θ`, `v^± = −c sin θ/√2` | 1 |
| vector | +1 | `(α/√2)(x̂ + iŷ)`: `v^0 = (α/√2) sin θ`, `v^- = (α/2)(cos θ+1)`, `v^+ = (α/2)(cos θ−1)` | 1 |
| vector | −1 | as +1 with `v^- ↔ v^+` | 1 |
| vector | \|m\| ≥ 2 | 0 | 0 |

The vector rules follow from `v_z ~ e^{imφ}`, `v_x ± i v_y ~ e^{i(m±1)φ}`
(spin basis `v^± = (v_θ ± i v_φ)/√2`). The shape functions `s_σ(θ)`
satisfy `Σ_σ s_σ² = 1`, and the Cartesian projection that isolates the
DOF is the same combination: `v_z = Σ s_σ v^σ` (m = 0),
`v_α = (x̂ − iŷ)·v/√2 = Σ s_σ v^σ` (m = +1).

Near the origin an m = 0 scalar is `f(0) + a z + b(x²+y²) + c z² + …`, so
`∂_r f(0, θ) = a cos θ` and `f_rr(0, θ) = 2b sin²θ + 2c cos²θ`; hence
`Lap f(0) = 4b + 2c = 3 <f_rr(0, ·)>_{l=0}`.

## Operator limits: symbolic L'Hospital

Each builder assembles `r^p · op` from a closed-form spin expression
polynomial in r. Substituting `v^σ(r, θ) = Σ_k r^k V_k^σ(θ)`
(`V_k = ∂_r^k v(0, θ)/k!`) and taking the coefficient of `r^p` gives the
limit of `op(v)` at r = 0 as a combination of the `V_k(θ)` and their
θ-derivatives; e.g. `r²·Lap f → 6 V_2 + ∇_H² V_2 = 4b + 2c`. The
`V_k(θ_j)` are one-sided k-th derivatives at r = 0 over `order_r + k`
nodes at the actual node positions (non-uniform grids supported), the
θ-derivatives use the bundle's stencils. Backgrounds inside the
coefficients (advection, `Dot(bg, tensor)`) are expanded the same way
from their grid values. The composed MHD blocks `X·C`, `C·X` inherit the
Curl rows because `X` is pointwise. `LapH` is not Cartesian-regular and
keeps the angular operator on `f(0, ·)`.

These rows replace the i_r = 0 rows of every differential block in the
builders (`install_origin_rows`), the mass conversion at i_r = 0 is
`1/k_op`, and the masses carry identity placeholders there, so the
i_r = 0 rows of `A_total`/`B_total`/`b_total` hold the unscaled equation
at the origin including coefficient, sign, pointwise, identity, Dt and
source terms.

A row built from the interior rows of the same block (extrapolation of
`block(r_i)/r_i^p`, or a derivative of the r^p-scaled block) is a linear
combination of the interior equations: it leaves the origin DOF
undetermined (`Div(Grad(v))` BVP: smallest singular value 7e-6, O(1)
solution error). Origin rows must use node-0 stencils of the unknown.

## Row replacement in `CompiledProblem`

For every primary unknown, in the i_r = 0 rows of its paired equation
(`paired_equation_of_unknown`, so the placement is right in both the
equation-order and the alphabetical layout):

- one DOF: anchor at `(σ_a, i_r=0, j=0)` = `Σ_j w_j Σ_σ s_σ(θ_j) row(σ, 0, j)`
  applied to K, to M and to b (`w` = l = 0 Legendre projection weights,
  exact for polynomials in cos θ, condition number < 20 up to N_θ ≈ 100);
  ties `v^σ(0, θ_j) − (s_σ(θ_j)/s_σa(θ_0)) v^σa(0, θ_0) = 0` elsewhere,
  zero in M;
- no DOF: Dirichlet-zero rows, zero in M.
- a tensor unknown (nine covariant spin components `T_{σ_a σ_b}`, each
  an (N_r, N_θ) block): its origin values at mode m span the space of a
  constant Cartesian matrix's e^{imφ} content, dimension 3 (m = 0:
  the l = 0, 1, 2 parts), 2 (|m| = 1), 1 (|m| = 2) and 0 (|m| ≥ 3);
  `ball_origin.tensor_origin_basis` derives it symbolically (the
  e^{imφ} coefficient of `e_σa · E_ij · e_σb` over the nine Cartesian
  unit tensors, `e_0 = r̂`, `e_± = (θ̂ ∓ i φ̂)/√2`, orthonormalised on the
  θ grid; the same derivation reproduces the vector shapes above). The
  d anchors sit at the d best-conditioned positions (pivoted QR of the
  basis) and carry the basis projections `Σ_j w_j Σ_ab s^(k)_ab(θ_j)
  row(ab, 0, j)` of the equation in K, M and b; every other origin
  position q is tied to the anchors, `T(q) − Σ_k C[k, q] T(p_k) = 0`
  with `C = S_anchor⁻¹ S`, zero in M
  (`CompiledProblem._apply_tensor_origin_regularity`).

Every unknown is treated, whether or not a second-order operator acts on
it: the m = 0 pressure gets its ties and the anchor `Div v(0) = 0`, the
|m| = 1 inviscid velocity its α-anchor. The gauge row of an m = 0 scalar
sits at `(i_r = N_r−2, j = 0)` of its paired equation on a ball
(`(1, 0)` on a shell), the first interior node inside the outer wall, so
it never meets an origin row or a wall row; at m ≠ 0 no gauge row is
installed (no constant mode exists; the row `Σ w p_m = 0` corrupted v
and p at O(1)). Quadrature weights are trapezoid on the actual radial
nodes and midpoint cells in θ (bit-identical to the old closed forms on
uniform grids).

## Boundary rows belong to equations

Every row of `A_total` belongs to an equation, and a boundary condition
replaces rows of the equation paired with its unknown
(`CompiledProblem._bc_host`), whatever the column layout: a scalar BC
the wall row of the scalar's equation, a vector projection the wall
row of the spin block that carries its component (θ̂ → σ = +1, φ̂ →
σ = −1, r̂ → σ = 0), a tensor-derived scalar BC (stress-free) its
metadata host, a projected curl / cross combination (the tangential
electric field `η Curl(b) − Cross(v, B0)` of a conducting wall,
`BC_PROJECTION_CURL_CROSS`) the spin block of its unit on the unknown
under the Curl, and a composite row (`BC_LINEAR_COMBINATION`: Robin,
Biot, Navier slip, the normal stress) the `host_unknown` /
`host_sigma` of its record. Formerly the rows were written at the unknown's column
offset, which in the alphabetical (no-`Dt`) layout of a v-p system
landed `r̂·v` on the θ-momentum wall row and `θ̂·v` on the continuity
wall row; that accident is gone and both layouts now assemble the same
rows.

At a wall node of a v-p system there are four equations, one
impenetrability condition and four unknowns, so one equation row gives
way. Which one is `impenetrability_host` (`compile_problem`):

- `'constraint'`: `Dot(r̂, v) = 0` replaces the wall row of the
  divergence constraint of v (`CompiledMatrices.
  divergence_constraint_of_vector`: the scalar equation with a `Div(v)`
  task, no `Dt` and no other unknown), and the wall-normal momentum row
  stays alive. On an impenetrable wall the normal momentum equation is
  the pressure's boundary condition, `∂_n p = n̂·(F − Dt v)`, so no
  Neumann row is written for p; its one-sided `∂_r p` sees the
  r-sawtooth that the centred interior stencils annihilate, and the
  m = 0 pressure checkerboard (a p-only null vector of K, smallest
  singular value 4e-15 with the momentum host) is removed (0.03 after,
  the constant mode pinned by the gauge row). The Stokes BVP
  `Lap(v) + Grad(p) = Grad(r²)`, no-slip, m = 0 then returns p to
  round-off (1.5e-11 at N = 24; 7e+2 with the checkerboard). A
  Dirichlet pressure at the same wall claims the constraint row first,
  and the projection falls back to the momentum row.
- `'momentum'`: the textbook closure, `Dot(r̂, v) = 0` on the σ = 0 wall
  row of v's own equation and the constraint kept at the wall.
- `'auto'` (default): `'constraint'` at m = 0 and `'momentum'` at
  m ≠ 0. At m ≠ 0 the azimuthal gradient `i m p/(r sin θ)` in the
  tangential momentum rows fixes the wall pressure pointwise, so the
  pressure needs no natural condition there, and on the first-order
  (v_r, p) system of an inviscid problem keeping the constraint at the
  wall is the far more accurate closure: the Zhang targets on 24×48 at
  order 6 (twin-grid filter) have relative errors 1e-6 to 7e-6 with the
  momentum host against 3e-5 to 4e-4 with the constraint host (and
  2e-6 to 9e-4 with the former θ-momentum accident), e.g. m = 1
  ω = 1.86168: 1.8e-6 / 1.5e-4 / 1.9e-5, m = 2 ω = 1.49644:
  1.3e-6 / 4.4e-4 / 3.2e-5; the constraint host converges at the
  scheme's order only from 24×48 on (1.5e-5 at 32×64). The solar-eigenproblem
  Rossby pencil (shell 0.71–0.985, m = 2) moves the same way: 6.2e-4 /
  2.9e-3 at 24×72, 2.2e-4 / 6.1e-4 at 32×96.

Both hosts leave the m ≠ 0 pressure columns of K non-singular
(smallest singular value of `K[:, p]` above 1). With the constraint host
the inviscid K itself is exactly singular: `Ω·(momentum row)` cancels
the Coriolis term and leaves `∂_z p`, whose square FD matrix has the
constants in its kernel and hence a left null vector, and with every
momentum row kept that left null vector transfers to K. Its right null
vectors are v-sawtooth modes with `M x ≠ 0`, i.e. spurious eigenvalues
at exactly λ = 0 (3–6 of them at N_r = 8–16); with the momentum host
the same modes sit at |λ| ~ 1e-4–1e-3 instead. Neither placement puts
anything near the inertial targets.

## Composite boundary rows

`compile_problem` walks the boundary conditions with `composite=True`
and binds the parameters into them (`bind_bc_parameters`). Each
single-shape kind has a row builder (`CompiledProblem._term_rows`,
per wall node a {column: coefficient} dict): the wall value of a
scalar, the one-sided `∂_r` over `order_r + 1` nodes, the unit
projection of a vector, the discretised `scalar_expr` of a tensor
projection, and the curl / cross combination projected on its unit
(the spin-basis curl formulas of `curl_cross_sparse` divided by r and
discretised like a tensor projection, the Cross part pointwise in the
background's spin components at the wall nodes). A
`BC_LINEAR_COMBINATION` row is `Σ_i c_i(r_wall, θ_j) · row_i`, the
coefficients evaluated at every wall node (θ dependence allowed), on
the host the record names; a single-shape row is installed exactly as
before, byte for byte on the certified pencils. Closed forms
(`tests/test_collocated_parity.py`, order 4, 12×18 → 18×27):

- Robin `dot(r_hat, grad(p)) + h p = 0` on the unit ball,
  `k j_l'(k) + h j_l(k) = 0`, m = 0, 1, 2, h = 1, 3: rms error
  2.9e-4–1.1e-3 at 18×27, rms order 4.2–5.0; the shell analogue with
  the outward-normal condition at both walls (`− h p` at r = r_i), the
  2×2 Bessel determinant of `a j_l + b y_l`, rms error 1.1e-4–7.3e-4,
  order 3.8–4.3; the Biot wall with symbolic κ and Bi binds at compile
  time, and every spelling of the coefficient gives the same pencil.
- Navier slip, two composite rows on σ = +1 and σ = −1 of v:
  `(1 − l_s) j_l(k) + l_s k j_l'(k) = 0`, errors 3–4e-4 at 18×27.
- The traction-free Stokes ball, `dot(r_hat, dot(strain, r_hat)) − c p
  = 0` on the σ = 0 wall row of v with columns of v and p (linear in
  c: the `−c p` and `+c p` pencils average to the row without p): the
  toroidal family `k j_l'(k) = j_l(k)` and the poloidal roots of the
  determinant of `tests/test_dedalus_ball.py` (l = 2, c = 2: 8,
  28.211083; l = 1: 14.978747) to 1e-5–3.8e-3 at 18×27, rms order
  3.4–4.8. The l = 0 root λ = 10/c of the Dedalus test is an
  artefact of its gauge tau (`div(v) + τ = 0` admits `v = α r`) and has
  no counterpart here, where `div(v) = 0` holds at every node.

## Tensor unknowns and definitions

A rank-2 unknown is stored in its nine covariant spin components
(`tensor_unknown_sparse`, the components the reducer produces for a
tensor expression, so a definition is componentwise): the tensor
equation's rows are 9·N_r·N_θ, `TASK_TENSOR_OF_VECTOR` is the
(9N, 3N) block of the walker's `r T_{σ_a σ_b}` (r-power 1, the tensor
equation's mass `diag(r)` on the nine blocks), `TASK_DIV_TENSOR_UNK`
the (3N, 9N) block of `r² (Div T)^σ` from the `U = r T` form of
`r2_div_tensor_spin_components` (r-power 2), `TASK_DOT_BG_VEC_UNK_TENSOR`
the pointwise contraction `Σ_σin a^σin T_{σin, −σout}` (or
`T_{−σout, σin} a^σin` for `Dot(T, B0)`), identity and Dt on the nine
blocks, a background tensor source in covariant components. On a ball
the origin rows of the two differential blocks are the symbolic
L'Hospital limits with the index pairs as the spin labels; applied to
the nodes of a Cartesian-smooth `u` and of `grad(u)` the blocks
converge at order 4.1–4.9 on shell and ball at m = 0, 1, 2, the origin
rows to the r → 0 limits.

A definition `T − tensor(u) = 0` (class 'definition' of
`spdee.ir.constraints.classify`, consulted whenever the system has a
tensor unknown) is eliminated at compile time by rewriting the tasks:
`r T_{σ_a σ_b} = −(1/c) Σ_k r T_k[σ_a σ_b]` for the identity coefficient
c (SymPy) and the walker's r-scaled components of the tensor terms, so
`Div(T)` becomes the closed-form `TASK_DIV_TENSOR_VECTOR` on u and
`Dot(B0, T)` the `TASK_DOT_BG_VEC_TENSOR_UNK` contraction; T and its
equation leave the layout (`CompiledMatrices.eliminated_tensors`). The
eliminated pencil of `dt(v) − Ek div(T) + grad(p) = 0, T − grad(v) −
trans(grad(v)) = 0, div(v) = 0` is entrywise the strain-form pencil
(difference 0.0 in A and B) with the same spectrum, on shell and ball.
With `dt(T)` the tensor keeps its nine blocks (13 blocks in the layout,
B = `diag(r)` on them, d = 3, 2, 1, 0 origin anchors with mass at
|m| = 0, 1, 2, ≥ 3 on the ball).

## Involutions: divergence cleaning

A written `Div(b) = 0` with no scalar left to host it is put to the
classifier (`classify(..., mimetic=False, core="collocated")`); an
involution (a curl evolution, no multiplier) takes its
`DivergenceCleaning`: the scalar `__div_b` joins the roster (rank 0),
`+ Grad(__div_b)` joins b's evolution equation as a `TASK_GRAD_SCALAR`,
the constraint is paired with the multiplier, the multiplier is
Dirichlet zero at the walls of the record (`CompiledMatrices.
implicit_bc_tasks`, appended by `compile_problem`) and mean-zero gauged
at m = 0. With the multiplier's wall rows on the constraint, `r̂·b = 0`
hosts on the σ = 0 momentum row (`_hosts_on_constraint`). Free decay
in a perfectly conducting shell at m = 0 (the toroidal l = 1 rates from
`d(rT)/dr = 0`, k = 1.984570 and 6.618465; the poloidal l = 1, 2 from
S = 0, k = 6.572013 and 7.111576) lands to 3e-5–5e-4 at 18×27, rms order
3.7; on the ball (k = 2.743707, 6.116764; 4.493409, 5.763459) to
3e-5–3e-4, rms order 2.9 with the coarse error already near the order
floor. The eigenvectors satisfy the interior constraint rows to
round-off and the rows the multiplier's walls, the origin ties and the
gauge replace to the discretisation error (4.8e-2 → 1.2e-2 on the
shell). The conducting wall alone (`b_free`, the tangential `curl(b)`
rows) gives the same rates. An implied involution (an induction
equation with no written constraint, `add_magnetic_field`'s default)
is left as before, without a constraint: the MHD pencils of the
existing sweeps are unchanged.

## The vector potential

`TASK_CROSS_CURL_CURL_UNK_BG`, the Lorentz force `Cross(Curl(Curl(A)),
B0)` of the A formulation, is `X @ CC`
(`curl_cross_sparse.build_cross_curl_curl_unk_bg_block`): the pointwise
Cross block times the r² Curl(Curl) block, r-power 2, the origin rows
inherited from CC. The A formulation of the free decay (Coulomb gauge
as a saddle point on ψ, tangential A = 0 and ψ = 0) gives the b
formulation's rates (shell rms order 3.5, gaps below 1e-2). The 2 kG
dipole on the m = 8 Figure-2 shell of the solar eigenproblem shifts the sectoral Rossby
mode by −0.0695−0.0107j (b, conducting walls through the tangential
electric-field rows) and −0.0673−0.0090j (A) at 12×24, −0.0698−0.0106j
and −0.0683−0.0095j at 16×32 (order 4; the Dedalus values are
−0.0676−0.0091j and −0.0686−0.0102j, the staggered −0.0664−0.0084j), the
two formulations approaching each other as the grid is refined.

## Curl(Curl(v))

`curl_cross_sparse.build_r2_curl_curl_vector_block` composes the
closed-form Curl spin formula with itself: with `W^σ = r·(Curl v)^σ`,
`r²·(Curl Curl v)^σ = r·[r·Curl(W/r)]^σ`, which is polynomial in r and
equals `r²·Grad(Div v) − r²·Lap(v)` term by term, so the assembled block
(r-power 2, mixed `∂_r∂_θ` through the product stencil, origin rows from
the symbolic L'Hospital rule) is the difference of the Grad(Div) and Lap
blocks entrywise to round-off — interior, walls and origin rows alike
(`tests/test_curl_cross_new.py::test_collocated_core_builds_curl_of_curl`,
relative difference below 3e-16). The coverage cells pass at order
4.0–4.8 (shell) and 4.4–5.3 (ball).

## A background on the right of a Dot

The walker canonicalises `Dot(X, B0) = Dot(B0, Transpose(X))` for a
tensor expression X of an unknown vector (exact: `X_ij B0_j =
(Xᵀ)_ji B0_j`; the reduced spin components and the Grad/Transpose(Grad)
weights are swapped) and `Dot(v, T0) = Dot(Transpose(T0), v)` with the
background tensor transposed component-wise (a symmetric T0 passes
through unchanged). Both emit the existing task kinds, so every core
that builds the left form builds the right one: the collocated cells
pass at the orders of their mirrors, the staggered core inherits its
mirrors' verdicts (`Dot(vector, T0)` on a shell passes, on a ball
fails at |m| = 1, 2 like `Dot(T0, vector)`; `Dot(T(vector), B0)` is
refused with the radial-background message of `Dot(B0, T(vector))`).

## Benchmarks

`tests/test_ball_collocated.py` runs each case at two resolutions and
asserts the measured order; the table in the test log lists analytic
value, computed value, error and order for scalar Helmholtz (Dirichlet
and Neumann, m = 0, 1, 2, with coefficient and sign variants and as a
`Dt` eigenproblem), the manufactured Poisson BVP at orders 2, 4, 6 on a
non-uniform radial grid, stress-free toroidal decay modes
(`k j_l'(k) = j_l(k)`, m = 0, 1, 2, divergence-free eigenvectors),
inviscid inertial modes (Zhang et al. 2001 targets at m = 1, 2) and the
gauge mechanics.

### The three m = 2 Zhang targets of the dense census

`docs/analytic_benchmarks.md` (finding 4) records that the census of
`tests/test_analytic_evp.py` (order 4, 12×18 → 18×27, nearest neutral
eigenvalue) does not converge for ω_Z = 1.64344, 1.15306, 1.73397 at
m = 2. Identified by eigenvector instead — the pressure of a Zhang
(N, m, parity) mode is `s^m e^{imφ}` times a polynomial in (s², z) of
degree 2N (SYM) or 2N + 1 (ANTI), and the weighted least-squares
residual of a computed p on that space is the identification
(`tests/test_evp_zhang_2001.py::test_m2_slow_targets_identified_by_pressure_polynomial`)
— the picture is:

- at the census resolutions no neutral eigenvector near the three
  targets lies in its polynomial space (residuals 0.10–0.32 against
  1e-3–1e-2 for the seven converging targets): the modes are not there
  as neutral modes. Each has collided with a neighbouring band mode
  into a complex pair, |Im ω| = 8.7e-3, 2.5e-2, 1.9e-2 at 12×18 and
  7.9e-3, 2.1e-3, 1.4e-2 at 18×27 (the non-adjoint collocated
  Grad/Div; the pairs are the "non-neutral spurious modes" of
  `tests/test_checkerboard_probe.py` landing on a target). The
  neutral gate `|Im ω| < 1e-6` drops them and the nearest survivor is
  a band neighbour at the local spacing (2–5e-3), whose distance does
  not fall with N: the census measures the spacing, not the mode.
- the real parts of the pairs converge: 2.3e-4, 4.0e-4, 2.8e-4 at
  24×48 order 4 (Im 2.1e-3, 1.1e-3, 7.7e-4), and at order 6 on 24×48
  all three are neutral to 1e-15, in their polynomial space to
  4e-3–1e-2 and within 2.2e-5, 3.7e-5, 5.3e-5 of the targets (the
  twin-grid tests of `tests/test_evp_zhang_2001.py`).

So it is the core at those resolutions — the collocated pencil is not
skew and the collision of a target with a band mode is a resolution
accident that closes with N — and the census reads it as a
misidentification. A census that wants these three at 12×18–18×27
has to identify by eigenvector and read the real part of the pair, or
exclude them from the order.

## Known limitations (not origin-related)

- The collocated Grad/Div pair is not discretely adjoint: inviscid
  inertial eigenvalues of under-resolved modes carry damping
  `|Re λ|` up to 1e-2 that shrinks with resolution; converged modes are
  neutral to 1e-12 (`tests/test_checkerboard_probe.py`).
- The m = 0 pressure of a v-p system with the textbook wall closure
  (`impenetrability_host='momentum'`) has the collocated checkerboard
  null mode (a 0/0 pencil singularity in an EVP, shift-invert then
  reports eigenvalues that track the shift; an arbitrary sawtooth in a
  BVP), on shells and balls alike; the default `'auto'` hosts the
  impenetrability condition on the continuity wall row at m = 0, which
  removes it (section above).
