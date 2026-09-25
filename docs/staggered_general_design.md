# General staggered (Arakawa-C) assembler with differential-algebraic constraints

Design of the general staggered assembler, which consumes the IR walker's task list the way
the collocated compiler does and classifies and handles the algebraic constraints a
string-entered system may carry.  It replaced the pattern-based compile path
(`_detect_velocity_pressure`, `_parse_incompressible_pair`, `_assemble_compressible_cgrid`,
`recognize_mhd`, `extend_with_mhd`), deleted in stage 5; section 7 records what went and
what the acceptance identities were.

## 1. Scope

Any linear system of the IR's operator vocabulary whose unknowns are scalars, vectors and
rank-2 tensors (section 2b places a tensor component by component at the strain
locations).  Backgrounds may be scalars, vectors and tensors, as today.

The physics this must cover, each of which couples an evolution equation to a constraint:

| system | evolution | constraint | multiplier |
|---|---|---|---|
| inviscid / viscous rotating flow | Dt(v) + ... + Grad(p) | Div(v) = 0 | p |
| Boussinesq convection | Dt(v) + ..., Dt(T) + ... | Div(v) = 0 | p |
| anelastic | Dt(v) + ..., Dt(s) + ... | Div(rho0 v) = 0 | p (or reduced pressure) |
| fully compressible (Bekki) | Dt(v), Dt(rho1), Dt(s1) | none (continuity is evolutionary) | none |
| induction, b formulation | Dt(b) = Curl(Cross(v, B0)) + eta Lap(b) | Div(b) = 0 | none: an involution |
| induction, A formulation | Dt(A) = ... - Grad(psi) | Div(A) = 0 (Coulomb gauge) | psi |

## 2. Placement on the C-grid

| unknown kind | location | DOF blocks |
|---|---|---|
| scalar (p, T, s, rho1, psi) | cell centres (r_c, theta_c) | 1 |
| vector, "primal" (v, A, any vector whose equation carries Grad of a scalar) | faces: v_r on (r_e, theta_c), v_theta on (r_c, theta_e), v_phi on (r_c, theta_c) | 3 |
| vector, "flux" (b when its constraint is an involution) | the same face placement, read as face fluxes (constrained transport) | 3 |

The single vector placement keeps every vector unknown on the same faces, so the divergence
of any vector lands on the centres and the curl of any vector lands on the dual edges
(`build_staggered_curl`); the CT induction path already uses this reading
(`CTGrid`).  Edge quantities (electric field, circulations of A) never become unknowns: they
are intermediate images of the operators.

## 2a. Placement as grade inference (the general rule behind section 2)

The C-grid is a discrete de Rham complex: cell centres carry 0-forms (or, dually,
3-forms: cell volumes), r/theta/phi edges carry 1-forms (circulations), faces carry
2-forms (fluxes).  Grad maps centres to faces (0 -> 1 on the dual grid), Curl maps faces to
edges, Div maps faces to centres, and the identities Curl Grad = 0 and Div Curl = 0 hold
exactly.  Every unknown must be given a grade so that each equation is homogeneous in grade
and the constraints it carries land at the grade of their multiplier or close by identity:

- a scalar unknown is a 0/3-form (centres);
- a vector unknown whose constraint is Div(u) = 0 (or Div(M u) = 0) with a multiplier under
  Grad is a face field: the constraint lands on the centres, where the multiplier lives, and
  the pair (Grad, -Div) is adjoint;
- a vector unknown whose evolution is Dt(u) = Curl(...) and whose constraint is Div(u) = 0
  without a multiplier is a face field read as fluxes: Div Curl = 0 closes the constraint by
  identity (constrained transport);
- a vector unknown u whose constraint is Div(u) = 0 with a multiplier and whose curl is
  differentiated again (Curl Curl u, the vector potential) is a face field too: Curl u lands
  on the edges and the second Curl returns to the faces; the same placement as the velocity,
  so the products Cross(v, B0) and Cross(Curl(Curl(A)), B0) meet on the faces.

Section 2's "every vector on the faces" is the outcome of this rule for all the systems in
scope; the assembler stores the grade per unknown (a `placement` attribute, "center" or
"face") so that an edge placement can be added later without touching the dispatcher.
Products that change grade (Dot and Cross with a background, the Coriolis pairs) are
discrete Hodge maps: an interpolation P and its norm adjoint W^-1 P^T W, used as a pair so
that the assembled operator inherits the skewness or self-adjointness of the continuous one.
The pair is between the two blocks that exchange energy, which is within one equation for
Coriolis (the two off-diagonal velocity blocks) and across two equations for the ideal MHD
coupling (induction against Lorentz, kinetic against magnetic energy).  The cross-equation
case is not achieved for constrained transport: building the Lorentz maps as the norm
adjoints of the induction interpolations does not make the ideal pencil skew, so the CT
recipe's plain averages stand (section 8); why the flux/area conversion defeats the naive
adjoint is open.  The constraint classification of section 4 is then a property of the grade
graph: a constraint that is the image of an exact identity is an involution; a constraint
that lands on a multiplier's grade is a saddle point; anything else has no consistent grade
and is refused.

## 2b. Tensors

A rank-2 tensor is placed component by component where the corresponding velocity
gradient is centred, which is what `build_staggered_strain` already does for the strain:

| component | location | grade reading |
|---|---|---|
| T_rr, T_thth, T_phph | cell centres (r_c, theta_c) | derivative of a face component along its own direction |
| T_rth, T_thr | corners (r_e, theta_e) | cross derivative of the r and theta face components |
| T_rph, T_phr | r-faces (r_e, theta_c) | the v_r location |
| T_thph, T_phth | theta-faces (r_c, theta_e) | the v_theta location |

With this placement the operators are exact maps of the complex: Grad(v) is the strain map
S (faces to the six locations, the symmetric part) or its unsymmetrised form with the
nine components; Div(T) is the norm adjoint -W_v^-1 S^T W_T back to the faces, which is
how the viscous Gram is built today; Transpose swaps components that share a location, so
it is a permutation; Trace lands on the centres; a scalar background multiplies pointwise
at each component's location; Dot(B0, T) and Dot(T, B0) are Hodge interpolations from the
component locations to the faces paired with their adjoints.  Lap of a tensor and Grad of
a tensor (rank 3) are refused.

A tensor unknown is therefore admissible with placement "tensor" (six or nine blocks): its
identity and Dt are diagonal on those blocks, and a first-order reduction such as
`T - Grad(v) = 0` classifies as an algebraic equation whose grade is that of T, i.e. a
definition rather than a constraint; the assembler eliminates it (T := Grad v) when T
appears nowhere under Dt, or keeps it on its own DOFs when it does.  Tensor
backgrounds are evaluated at the component locations and enter only through products.

Delivered (stage 3, `PLACEMENTS['tensor']` and `['tensor9']`): the six symmetric
components in the strain order rr, tt, pp (CC), rt (EE), rp (RC), tp (CE), and the nine
with tr (EE), pr (RC), pt (CE) appended.  Grad(u) + Transpose(Grad(u)) onto the six is
2S with S = `build_staggered_strain` (its wall closure and the |m| = 1 pole prolongation,
so Grad then Div is the strain Gram with the deviatoric map replaced by 2I, to rounding);
a Grad(u) / Transpose(Grad(u)) combination with a != b goes onto the nine through
`_CGridOps.grad9`, the unsymmetrised gradient.  Which
combination a tensor expression is, is read off its reduced components by least squares
on smooth trial fields (constants snapped to small rationals); a background tensor times
u or a background scalar inside the tensor expression is refused.  Div(T) is
-W_v^-1 M^T W_T with M the placement's gradient map and W_T its volume norm
(off-diagonals twice on the six), plus the pole share of `_strain_pole_share` (below);
for a tensor expression of a vector on the nine this divergence is not built:
`Div(a Grad(u) + b Transpose(Grad(u)))` with a != b is a Lap(u) + b Grad(Div(u)) on the
face blocks (section 3), and only a tensor unknown with DOFs carries the nine-component
divergence.  Transpose is the component permutation (the identity on the six).  Trace
sums the three CC diagonals and equals the norm-adjoint divergence only to O(h^2): the
flux form squares the edge radii where the strain averages v_r / r, so the two are
different consistent stencils of one identity, not the same matrix.

The nine-component map has two spellings of its connection terms (`grad9(wall_bc,
paired)`).  The placement spelling is the tensor placement's map, the one whose norm
adjoint is Div(T): v_r / r and cot(theta) v_theta / r on CC are the plain averages of
the strain, so tp + pt is 2 e_theta_phi entrywise; the wall rows of the radial shears rt,
rp carry the strain's closure (no-slip: the mirror ghost 2 v_tan / h reading the wall cell
only, the wall faces at half weight, the wall averages of tr, pr zero; stress-free:
d_r v_tan = v_tan / r from the wall cell), since the plain one-sided wall rows read the
second cell too and their adjoint leaks an O(1/h) term into it; and the pole rows of tp
and pt carry the parity limits of a regular field, `(1/r) d_theta u_phi` as
(1 - p) u_phi(theta_c) / (r dtheta) with p = (-1)^(|m|+1), and `-(cot theta / r) u_phi`,
`(i m / (r sin theta)) u_theta` by L'Hopital, so the pole DOFs of a tensor unknown hold
the limits.  The paired spelling is the map of the contraction Dot(B0, Grad v), which
takes no adjoint: the phi-row's connection terms are the Hodge pairs of the Coriolis
block with the metric factor on the pair's host grid (v_phi / r through CC -> RC,
cot(theta) v_phi / r as (1 / (r sin theta)) P(cos theta_c v_phi) through CC -> CE, and
v_r / r, cot(theta) v_theta / r through the adjoint maps RC -> CC, CE -> CC), so that
Dot(s phi_hat, Grad v) = i m v + z_hat x v holds entrywise against the 'center' Coriolis
pairing and the Doppler shift of a solid-body flow is exact to rounding
(`tests/test_analytic_evp.py`, shifts 3e-15).  The two differ where the adjoint averages
carry the Coriolis pair's O(1) value, on the pole-adjacent ring (v_theta ~ sin theta at
m = 0) and in the ball's first cell (v_r finite at the origin): harmless in a product
with a background that vanishes there, but O(1/dtheta) once divided, which is why the
placement keeps the plain averages.

The pole share.  (Div T)_phi carries (cot theta / r) T_phi_theta, whose norm adjoint of
the centres-to-edges average is the sin(theta)-weighted trapezoid over a cell's two
edges, (1 / (2 r sin theta_c)) sum_e cos(theta_e) T(e); at the pole-adjacent cell the
pole edge has zero weight and its share, T(pole) / (2 r sin theta_c) with cos theta = +-1,
is lost although cot(theta_e) W_e -> r^2 dr dtheta is finite there.  Wherever T_phi_theta
has a finite pole limit -- every m for a non-symmetric tensor, even |m| >= 2 for the
strain (e_theta_phi(pole) = (i m / 2r) (u_theta / sin theta)(0), the Hessian of
(x + i y)^2 is the example) -- the pole-adjacent v_phi row is O(1/dtheta) inconsistent
and the max-norm error is first order.  No weight or row of the pole edge repairs this
inside the norm adjoint: the u_phi row needs the pt pole DOF at the weight W_c / 2 with
its exact limit, whose (i m / (r sin theta)) u_theta part then feeds W_c / 2 times the
limit back into the v_theta ring, whose lumped weight already integrates the odd hat
function exactly (the flux form of tp needs no pole share at all, so a common weight
cannot serve both), and any u_phi-only stencil for the limit enters the NSD Gram with the
wrong sign.  `_strain_pole_share` therefore adds the share explicitly, reading the pole
DOF: always on the nine-component placement (uncertified; Div(T) of a non-symmetric
tensor unknown then converges at order 2 at m = 0, 1, 2 on shell and ball, 2.07 / 2.04 /
2.01 and 2.05 / 1.92 / 1.93 over N = 12, 24, 48), and on the six and in the strain Gram
behind `pole_closure=` (`compile_fd2_staggered`, `build_staggered_weighted_viscous_gram`,
`build_staggered_strain`: the pole rows of e_theta_phi carry the limit), off by default
because W F loses Hermiticity on that ring and the certified m = 8 pencil moves
(section 3b).

Dot(B0, T) and Dot(T, B0) with any B0(r, theta) (`_CGridOps.dot_vector`): every
component T_ij is multiplied by B0_i (or B0_j) evaluated at the component's own location
and brought to the face of the free index by the Hodge map from there (CC -> RC, EE -> CE,
RC -> CC and their partners), so the block is the interpolation-paired product of section
2a; a component of B0 that vanishes identically contributes no block, and the radial case
is `dot_radial`, the same maps with a unit background.  On a ball the origin ring rows of
the contraction are the guarded zero (the phi-row components carry 1/r and B0_phi
vanishes there), so a viscous fold at m = 0, |m| = 1 carries the Coriolis pair's ring
entries but not the advection's; the inviscid anchor row is receive-only and decoupled.
Against the hand `_build_staggered_gradlnrho_strain` blocks (centred same-grid
derivatives, row-normalised averages; never part of the certified assembler)
Dot(r_hat, Grad v + Grad v^T) agrees to second order on smooth fields, not entrywise.

## 3. Operators: task kind to mimetic block

The walker (`spdee.ir.walker.walk_equation_system`) emits one `OperatorTask` per term with
`kind`, `equation_name`, `unknown_name`, `sign`, `coefficient` (a scalar background
prefactor) and `source_field` (the background operand).  The staggered dispatcher maps each
kind to a block between the unknown's DOFs and the equation's rows; the equation's rows live
where its principal unknown lives (the unknown under `Dt`, or for a constraint the unknown the
constraint is assigned to, section 4).

| task kind | block | pairing rule | stage-2 builder (`staggered_general`) |
|---|---|---|---|
| identity (scalar, vector, tensor) | diagonal on the unknown's own DOFs | coefficient evaluated at the DOF location | `_Assembly.identity`: `sp.eye` per placement block, coefficient on each block's sub-grid, A or B by target; on an eliminated tensor the block is composed with its definition |
| grad_scalar | G: centres to faces (`_build_staggered_gradient_blocks`) | primal | `_CGridOps.gradient` (the three blocks stacked in face order), coefficient on the faces; Grad(M s), which the walker distributes over sums of weighted unknown scalars and expands into a tagged product-rule pair (`grad_weight`), is built from the pair as G diag(M) with M on the centres (the seed's spelling of the pressure gradient) |
| div_vector_scalar | Div = -W_c^-1 G^T W_v (norm adjoint of G) (`build_staggered_grad_div_pair`) | adjoint of grad, so the saddle-point pair is exactly skew | `_CGridOps.divergence(M)`: -W_c^-1 G^T (M W_v); the walker expands Div(M u) into a tagged task pair (`div_weight`) that the dispatcher consumes as one block; a constraint row may carry only a constant scale |
| lap_scalar | Div Grad with the flux stencils (`_radial_flux_stencils`, wall closure from the BC) | self-adjoint, NSD | `build_staggered_scalar_laplacian(wall_bc=(inner, outer))`, closures from the scalar's BC tasks (natural Neumann without one); with a `dot_bg_vec_grad_scalar` of the same equation whose background is radial and whose coefficient is a radial multiple a_r of the (constant) Lap coefficient, the pair is the w-weighted Laplacian `build_staggered_weighted_scalar_laplacian` with w = exp(integral a_r dr) by the trapezoid rule on the union radii (the certified P0 recovery of the hand assembler, bitwise) |
| lap_vector | the strain Gram on the constraint manifold, the true vector Laplacian off it | self-adjoint, NSD | `build_staggered_weighted_viscous_gram(wall_bc)` (= Lap + (1/3) Grad Div, i.e. Lap on the constraint manifold), the closure from the vector's tangential BC tasks; the Gram stands in for Lap(u) only where the two agree -- an unknown that carries a divergence constraint (`record['constraints']`, saddle point or involution) or an equation that writes Lap(u) + (1/3) Grad(Div(u)) itself, which is the compressible viscous force -- and otherwise Lap(u) is the Gram less (1/3) `build_staggered_grad_div` (section 3a); read together with the equation's grad_div, symmetric-strain and trace terms on the same unknown (`_Assembly.viscous_force`): with a . (Grad u + Grad u^T) - (2/3) a Div(u), a radial and proportional to the Lap coefficient, the Gram is density-weighted with w = exp(integral a_r dr) recovered by adaptive quadrature (matches the hand assembler's rho0 to rounding); otherwise the extra terms are built on their own |
| grad_div_vector | G Div | part of the viscous Gram when present | `build_staggered_grad_div` scaled by (c_g - c_l / 3) on top of the Gram, so Lap + (1/3) Grad Div is one Gram (the compressible force) and Lap + Grad Div is Gram + (2/3) Grad Div; on its own, c_g G Div |
| curl_vector | faces to edges (`build_staggered_curl`) then back to the faces by the Hodge maps | the edge curl of the primal complex | `_Assembly.curl_vector`: `_CGridOps.curl_to_faces`, the primal curl with the unknown's tangential ghost closure (the pole v_theta read through the |m| = 1 prolongation, section 3b), then hodge(CE, RC), hodge(RC, CE), hodge(EE, CC) |
| curl_curl_vector | Curl Curl u on the faces | the dual curl of the primal curl, so Div Curl = 0 is inherited | `_CGridOps.curl_curl` (`_Assembly.curl_curl_vector`); on an involution unknown it is dispatched to the CT resistive blocks instead (`involution_diffusion`), on a vector potential to `potential_diffusion` |
| cross_vector (Cross(B0, v), Coriolis) | interpolate the two partner components to the target face, multiply by the background component at the target face | the two off-diagonal blocks of each component pair are exact W-adjoints (the 'center' Coriolis pairing generalised: background evaluated on the centres of the pair, interpolations paired as P and W^-1 P^T W) | `_Assembly.cross_vector`: per component pair the third background component (times the coefficient) on the pair's host grid (CC for the pairs with v_phi, CE for (v_r, v_theta), so the ball origin ring only receives), blocks P D and -D W^-1 P^T W; verified W-skew to 2e-16 for b(r, theta) with all three components on shell and ball; 'seed' / 'adjoint' pairings kept for the certification |
| dot_bg_vec_unk_vec_scalar (Dot(B0, v) in a scalar equation) | faces to centres with the norm-adjoint of the centre-to-face interpolation, background at the centre | adjoint to bg_vec_times_unk_scalar | `_Assembly.dot_bg_vec_unk_vec_scalar`: [D_r hodge(RC,CC), D_t hodge(CE,CC), D_p] with the adjoint interpolations |
| bg_vec_times_unk_scalar (B0 s in a vector equation) | centres to faces, background at the face | adjoint partner of the previous | `_Assembly.bg_vec_times_unk_scalar`: [D_r hodge(CC,RC); D_t hodge(CC,CE); D_p]; the pair is exactly W-adjoint for constant backgrounds and to the interpolation error (first order entrywise, second order on smooth fields) otherwise |
| dot_bg_vec_grad_scalar (B0 . Grad s) | B0-weighted gradient on the centres | consistent with grad_scalar | `_Assembly.dot_bg_vec_grad_scalar`: the gradient blocks averaged to the centres (plain average: no partner block) and contracted with B0 there |
| dot_bg_grad_vec (B0 . Grad v) | the nine-component gradient map (paired spelling) contracted with B0 at the component locations, to the faces | the Hodge maps from each component location to the face of the free index; the phi-row reproduces the 'center' Coriolis pair, so Dot(s phi_hat, Grad v) = i m v + z_hat x v entrywise | `_Assembly.dot_bg_grad_vec`: `grad9(paired=True)` then `dot_vector` (section 2b) |
| dot_bg_vec_tensor_unk (B0 . tensor(v)) | the tensor map of section 2b, then the contraction | as above | `_Assembly.dot_bg_vec_tensor_unk`: (a, b) of a Grad(v) + b Transpose(Grad(v)) read off the reduced components, 2a S on the six when a = b (with the wall closure) else the paired nine-component map; consumed by the viscous combination when it fits (a radial B0 proportional to the Lap coefficient) |
| bg_vec_times_div_unk_vec | Div to centres, base interpolation to each face, B0 there | the Div is the norm adjoint of G | `_Assembly.bg_vec_times_div_unk_vec`; consumed by the viscous combination when it fits |
| div_tensor_vector (Div of a Grad(v) / Transpose(Grad(v)) combination) | symmetric: the strain map then the norm-adjoint divergence; otherwise a Lap + b Grad Div | Div(T) = -W_v^-1 M^T W_T, so Div(Grad v + Grad v^T) is the strain Gram with C = 2I; Div(Grad v) = Lap v and Div(Grad v^T) = Grad Div v for the rest | `_Assembly.div_tensor_vector`: the six-component route for a = b, else `lap_blocks` (the Gram less (1/3) Grad Div, and Grad Div); a background tensor times v is refused |
| dot_bg_tensor_vec (Dot(T0, v)) | diagonals of T0 on the faces, each off-diagonal pair through its host grid (CE for (r, theta), CC with phi) | base map one way, its adjoint the other, so a symmetric T0 gives a W-self-adjoint block | `_Assembly.dot_bg_tensor_vec` |
| identity_tensor, tensor_of_vector (a tensor equation) | the unknown's tensor blocks; the tensor map of the vector terms | the walker reduces the vector terms of a tensor equation as one symbolic sum (SymPy coefficients only) and reports its symmetry | `_Assembly.identity`, `_Assembly.tensor_of_vector` on a tensor with DOFs; a definition without Dt is eliminated (section 4) |
| div_tensor_unk, dot_bg_vec_unk_tensor (Div(T), Dot(B0, T), Dot(T, B0) of a tensor unknown) | the norm-adjoint divergence with the pole share / the Hodge-paired contraction of T's placement | as the vector cases | `_Assembly.div_tensor_unk`, `_Assembly.dot_bg_vec_unk_tensor`, composed with the definition when T is eliminated |
| curl_cross_unk_bg (Curl(Cross(v, B0)), induction) | the CT induction blocks (`ct_induction_blocks`) | div-free by construction (Div Curl = 0 on the fluxes) | `_Assembly.curl_cross_unk_bg`: E = c_E (v x B0) on the edges with B0 there, the wall circulations masked at a conducting wall, the incidence curl onto the face fluxes; a face unknown that is not an involution is divided by the face areas and its wall circulations follow v's wall (masked where v is no-slip) |
| cross_curl_unk_bg (Cross(Curl(b), B0), Lorentz) | the CT Lorentz blocks (`ct_lorentz_blocks`) | j = Curl b on the edges from the physical b = Phi / area, the plain two-point averages to the faces, B0 at the face (the seed's CT recipe, which the certification pins) | `_Assembly.cross_curl_unk_bg`, the coefficient restricted to the interior r- / theta-edges and the centres |
| cross_curl_curl_unk_bg (Cross(Curl(Curl(A)), B0), the vector potential's Lorentz force) | `curl_curl` then the Hodge-paired cross block | j returns to the faces before the product, so the pairing is that of cross_vector | `_Assembly.cross_curl_curl_unk_bg` |
| laph_scalar | angular part of the scalar Laplacian | self-adjoint | `build_staggered_angular_laplacian` (the theta stencils of the scalar Laplacian) |
| source terms (scalar, vector, tensor) | right-hand side at the equation's rows | | `_Assembly.source` into `StaggeredCompiledMatrices.b_total`, a tensor source evaluated component by component at the placement's locations |

After stage 4 every kind the walker emits has a C-grid block; `SUPPORTED_KINDS` is the
list and `general_path_reason` names any kind outside it.  What is still missing is
not a kind but a placement: an edge placement for an unknown (section 2a), and
`Lap`/`Grad` of a tensor (refused by the IR itself); constrained transport on a ball is
section 6.  Tasks whose coefficient vanishes
identically (Ek = 0 written into a viscous term, delta = 0) are dropped, so they neither
select the viscous wall reduction nor the r-power of the ball premultiply.

Everything a kind needs that does not exist yet is built as a new block builder in the
operator library; the dispatcher never contains physics.

## 3a. Lap of a vector: the Gram and the general operator

The strain Gram is `Div(2e - (2/3) Div(u) I) = Lap(u) + (1/3) Grad(Div(u))`, not `Lap(u)`.
The two agree exactly where `Div(u) = 0`, so the Gram is the right block for `Lap(u)` on a
face unknown that carries a divergence constraint (section 4: a saddle point or an
involution) -- which is where the certified entrywise identities live, the inertial
saddle point and the constrained-transport MHD pencil -- and it is the right block when
the equation writes `Lap(u) + (1/3) Grad(Div(u))` itself, which is the compressible
viscous force of the Bekki momentum equation.  Everywhere else it builds a different
operator from the one the user wrote.  Measured on a smooth non-solenoidal field
(shell, m = 2, interior rows): `|(Gram - Lap) u| / |Lap u|` = 2.33e-1, 2.31e-1, 2.30e-1
at N = 24, 48, 96, no convergence, while `|(Gram - Lap - GradDiv/3) u| / |Lap u|` =
1.76e-3, 4.02e-4, 1.87e-4.

`_Assembly.viscous_force` therefore reads the decision off the classification record and
not off equation names: the Gram when `record['constraints']` holds the unknown or when
the group's `Grad(Div(u))` coefficient is `c_l / 3` (or the density-weighted form of the
same combination), and otherwise the Gram less `(1/3) build_staggered_grad_div`, which is
the true vector Laplacian.  That spelling and not `Grad Div - Curl Curl`
(`build_staggered_vector_laplacian`): the latter is pointwise consistent in the interior
but its tangential wall rows grow like 1/h (0.96, 2.15, 4.56 at N = 12, 24, 48 against
the exact image), while the Gram carries the wall closure of the tangential BC and the
|m| = 1 pole prolongation and its difference with `G Div` inherits both.  In the coverage
matrix the shell cells of every vector operator then converge at order 2
(`docs/OPERATOR_COVERAGE.md`); what is left failing is the ball origin and the |m| = 1
pole of the curl, neither of which is the vector Laplacian.

## 3b. The pole rings

The theta-face DOFs of a face unknown at `theta = 0` and `theta = pi` are not degrees of
freedom: the face has zero area and zero volume weight, the strain Gram zeroes their rows
through the norm guard and reads the adjacent interior value instead at |m| = 1
(`_pole_vtheta_prolongation`), and the seed code does not carry them at all.  Left alone
they are an accident: `A_total` has 2 N_r rows that are empty or filled with the
1 / sin(theta) guard of a curl while `B_total` keeps the identity there, so the pencil
reads `0 = lambda x` and carries that many spurious zero eigenvalues, and a
boundary-value problem has no equation for them at all.  `_pole_face_rows` makes the pin
explicit -- a Dirichlet row, no mass -- and `compile_staggered_general` reports the
reduced-state indices as `constraints['pinned']`, which is the seed's convention
`v_theta(pole) = 0`.  At |m| = 1 the primal curl reads the pole DOF through the same
prolongation (`_CGridOps.primal_curl`): the pole closure of curl_r is the limit of
[(sin(theta) v_phi)' - i m v_theta] / sin(theta), whose numerator vanishes at the pole for a
regular field, and read from the pinned DOF it did not, an O(1) pole row that the dual curl
differentiated into O(1/dtheta) first theta-centre rows of Curl Curl.

At even |m| >= 2 the strain Gram's pole-adjacent v_phi rows are O(1/dtheta) inconsistent
(the pole share of section 2b): e_theta_phi has a finite pole limit there and the
sin(theta)-weighted average behind the cot term loses the pole edge.  Measured on the
manufactured Lap(vector) at m = 2 (shell, tau / B on the pole-adjacent v_phi ring): 10.4,
20.9, 41.7 at N = 24, 48, 96, and the max-norm error drops from order 2.02 to 1.12
between N = 24 and 48 (`docs/OPERATOR_COVERAGE.md` does not see it at N = 12, 24).  With
`pole_closure=True` the ring is 0.099, 0.049, 0.024 at N = 12, 24, 48 and the error
2.83e-2, 7.01e-3, 1.74e-3 (orders 2.02, 2.01; ball 1.98, 2.00).  The closure is inert at
m = 0 and |m| = 1 (the limit vanishes: entrywise the same Gram) and touches only the
pole-adjacent v_phi rows otherwise, but it is not a norm adjoint, so W F loses
Hermiticity on that ring and the certified m = 8 pencil moves: in
`scripts/certify_staggered_vs_seed.py --nr 12 --nt 48` with the closure the (vp, vt)
block's maximum grows from 2.982 to 3.379, its per-term viscous coefficient from
0.9994 to 1.2204 (the fit sees the new O(1/dtheta^2) entries) and the diffusive-group
residual of that block from 4.4e-4 to 1.5e-1 of the block maximum (pole rows 1.4e-1,
interior rows 1.6e-4, unchanged), while the sectoral Rossby mode (-0.219817-0.019127j),
its attribution table and the 611-mode dense window are the pinned ones to every printed
digit.  It stays off by default (the seed's own pole treatment is the weightless one), a
decision for the certification rather than for this document.  `Problem.compile` does not
forward the keyword; it is reached through `compile_fd2_staggered` /
`compile_staggered_general`.

## 4. Constraint classification

Each equation is classified by the task list it produced.

**Evolution equation.** Has a `TARGET_B` task on an unknown u.  Its rows live on u's DOFs
(the unknown's location) and B carries the identity on those DOFs.  A system with no `Dt`
anywhere is the eigenvalue operator itself, `L x = lambda x` (the Zhang spelling
`2 Cross(Omega, v) - Grad(p) = lambda v`): every vector equation whose only face unknown is
u is read as u's evolution equation with a unit `Dt(u)`, so B gets the identity on u's DOFs
(`_promote_eigenvalue_equations`, the record's `implicit_mass`).  A system that mixes `Dt`
and `Dt`-less equations keeps the algebraic classification below, so the reading is never a
guess about which equation carries the eigenvalue.

**Algebraic equation.** No `TARGET_B` task.  The assembler decides among:

1. **Saddle-point constraint.**  The equation is `Div(M u) = 0` (M a scalar background or 1)
   for a primal vector u, and some scalar unknown p appears in u's evolution equation only
   through `Grad(p)` or `Grad(c p)` and nowhere else with a `Dt`.  Then p is the multiplier:
   its DOFs are the centres, the constraint's rows are p's rows, B is zero there, and the
   Div block is the norm adjoint of the Grad block (with the same M weighting, so the pair
   stays adjoint under the rho0-weighted norm for the anelastic case).  At m = 0 the constant
   p is a null vector: the constraint row of one cell is replaced by the gauge p = 0, or by
   the integral gauge if the system declares p a gauge unknown.
2. **Involution.**  The equation is `Div(b) = 0` for a vector b whose evolution equation is
   `Dt(b) = Curl(E)` plus diffusion written as `Lap(b)` or `-Curl(Curl(b))`, with no scalar
   multiplier.  On the C-grid the discrete identity `Div Curl = 0` holds exactly, so the
   constraint is preserved by the evolution and adds no row; the constraint equation is
   dropped, the assembler records `constraints[b] = "involution"`, and the compiled object
   verifies `‖Div Curl‖ = 0` and reports the divergence of returned eigenvectors.  This is the
   CT path, and `Lap(b)` is rewritten as `-Curl Curl b` (+ Grad Div b, which vanishes on the
   constraint manifold) so the diffusion also preserves the involution.  The reading is also
   implied, without a written constraint, for an evolution whose A-terms are only curls of
   the state and its own diffusion (`add_magnetic_field` writes no constraint), with one
   exception (`_demote_pinned_involutions`): a vector that is the velocity of its own
   induction term, `Dt(u) = Curl(Cross(u, B0)) + ...`, and carries a no-slip or stress-free
   wall is a velocity, not a flux -- the tangential closure is its own -- and it is read as
   a plain evolution unknown (the vector Laplacian, the incidence curl divided by the face
   areas, the wall circulations masked where it is no-slip).  A field transported by another
   unknown keeps the involution reading, so pinned tangential components on it are still
   refused (section 5), as they are with a written `Div(b) = 0`.
3. **Gauge constraint.**  The equation is an integral (`integ(psi) = 0`) or a point value; the
   row replaces one row of psi's block, as today.
3b. **Tensor definition.**  A tensor equation without Dt whose terms are one unknown tensor
   T (with a nonvanishing coefficient) and tensor expressions of one vector unknown u: T is
   a definition, T := -(1/c_I) sum_k c_k tensor_k(u), eliminated from the state; every block
   that acts on T is composed with the definition map.  Its placement is the six symmetric
   components when the summed tensor expression is symmetric, else the nine.  With Dt(T)
   the equation is an evolution equation on the tensor DOFs (I on T, the tensor map on u).
4. **Unrecognised.**  Anything else (an algebraic relation between two unknowns without a
   multiplier, a constraint on a scalar, a divergence constraint on a vector that has no
   Grad of a multiplier and no Curl evolution) is refused with a message that states the
   equation, the classification attempted, and the two supported shapes.  This is the "detect
   and flag" requirement: the staggered core does not guess a discretisation for a constraint
   it cannot pair.

Index bookkeeping: the classification is a discrete version of the DAE index reduction.  A
saddle-point constraint is index 2 (the multiplier is fixed by one time derivative of the
constraint); the assembler does not reduce it but keeps the pencil singular in B on the
multiplier rows, which the shift-invert solvers handle; the involution is index 1 in the
sense that its time derivative is identically satisfied.

## 5. Boundary conditions

BC tasks come from `walk_boundary_conditions(system, m, composite=True)`
(docs/ir_boundary_and_constraints.md).  On the C-grid a BC is applied to the DOFs
that live on the wall (v_r on the wall r-edges) or to the closure of the stencils that reach
the wall (v_theta, v_phi and scalars, which live half a cell inside):

| BC task | staggered treatment |
|---|---|
| Dirichlet on the normal component (impenetrable, b_r = 0) | the wall v_r DOFs are eliminated symmetrically (`_eliminate_wall_velocity`) or overwritten inviscidly, as today |
| Dirichlet on tangential components (no-slip) | mirror-ghost closure in the strain / curl stencils reaching the wall (`build_staggered_strain(wall_bc='noslip')`), the slip closure below at ls = 0 |
| zero tangential traction (stress-free) | the natural condition of the strain Gram: the slip closure at ls = inf |
| Navier slip `Dot(t_hat, u) + ls Dot(t_hat, Dot(strain, r_hat)) = 0` | the slip closure with the slip length ls per wall node (`TangentialWall`) |
| scalar Dirichlet / Neumann | half-weight face closure in the flux stencils (`_radial_flux_stencils(wall_bc)`) |
| Robin `c_N Dot(r_hat, Grad(s)) + c_D s = 0` on a centre scalar | the mirror-ghost flux closure with the Robin coefficient (`_robin_wall_term`), order 2 |
| normal stress `Dot(r_hat, Dot(strain, r_hat)) - c p = 0` | the wall v_r faces stay DOFs (an open wall): the half-cell momentum rows of the strain Gram with the Dirichlet-closed gradient, plus the residual traction |
| tangential E = 0 / pinned b | the CT wall masks (`CTGrid.wall_masks`) |
| no condition at all on an involution unknown | the wall fluxes stay as DOFs and evolve with the unmasked wall circulations of the ideal electric field (kind 'free'); refused with a resistive term, whose wall current the one-sided corner stencils do not close |
| anything else (a composite row of another shape, tensor projections beyond the strain) | refused naming the term |

A BC that the walker cannot classify is refused rather than approximated.  The composite
rows reduce to the three closures below (`_composite_closure`); the coefficients are
evaluated at the wall nodes, so a theta-dependent coefficient is allowed.

**Robin.**  The row `c_N d_r s + c_D s = 0` at the wall face r_w, with the wall cell s_0 at
r_0 = r_w - sigma h/2 (sigma = +1 outer, -1 inner) and the mirror ghost s_g at
r_w + sigma h/2: the wall value is (s_0 + s_g)/2 and the wall derivative sigma (s_g - s_0)/h,
so the ghost is s_g = s_0 (2 c_N - sigma c_D h)/(2 c_N + sigma c_D h) and the wall flux is

    F_w = g s_0,    g = -c_D / (c_N + sigma c_D h / 2),

which is the Dirichlet closure -2 sigma s_0/h at c_N = 0 and the natural closure (no flux)
at c_D = 0.  The flux-form divergence counts the wall flux once, so the wall cell's row of
the Laplacian gains sigma r_w^2 g/(r_0^2 h) (times W(r_w)/W(r_0) under a weight), added to
the operator built with the natural stencils at that wall.  The closure is second order,
and, like the Dirichlet mirror ghost, only second order: a Robin relation is not a parity of
the field, so the reflected stencils of the higher orders do not carry it and the closure
is refused at order > 2.  Verified on the unit ball against k j_l'(k) + h j_l(k) = 0 and on
the shell against the two-wall determinant (`TestCompositeWallsBall`,
`TestCompositeWallsShell`).

**Navier slip.**  The row `v_t + ls (d_r v_t - v_t / r) = 0` (with the theta derivative of
v_r, which vanishes on a closed wall) is a Robin relation on the tangential velocity.  On
the shell the strain's radial shear is spelled r d_r(v_t/r) on the divided field g = v_t/r;
with the wall value (r_0 g_0 + r_g g_g)/2 and the shear r_w (g_g - g_0)/(r_g - r_0) the
ghost is

    g_g = gamma g_0,    gamma = (2 sigma ls r_w/h - r_0) / (2 sigma ls r_w/h + r_g),

so the wall row of d_r g is (1 - gamma)/(r_0 - r_g) g_0: at ls = 0 this is the no-slip
mirror g_g = -(r_0/r_g) g_0 (exactly, the same floating-point row), at ls = inf the even
reflection of g, whose row is empty.  On the ball the shear is spelled d_r v_t - v_t/r with
the two-point derivative and average, and the ghost is v_g = rho v_0 with

    rho = (sigma ls/h + ls/(2 r_w) - 1/2) / (sigma ls/h - ls/(2 r_w) + 1/2),

the derivative row (1 - rho)/(r_0 - r_g), the average (1 + rho)/2 (rho = -1 at ls = 0,
r_g/r_0 at ls = inf).  The tensor norm's wall shear faces carry the weight 1/2 + ls/h,
the certified 1/2 of the no-slip closure at ls = 0 and the once-counted stress at large ls;
the curl closures take the same ghost on the flux r v_t (one slip length along the wall).
The two limits are the existing 'noslip' and 'stress_free' pencils bitwise
(`test_navier_slip_limits_are_the_no_slip_and_stress_free_pencils`), and a no-slip wall
inside with a stress-free wall outside is the same closure with ls = (0, inf).  Verified on
the ball against (1 - ls) j_l(k) + ls k j_l'(k) = 0 at ls = 1/2.

**Normal stress.**  The row `2 c_T e_rr + c_p p = 0` on the multiplier p of `Div(u) = 0`
opens the wall: the wall v_r faces stay DOFs (`TangentialWall.open`), as the free wall of a
flux unknown keeps its fluxes, and their rows are the momentum balance of the half cell
inside the wall.  The strain Gram gives it with the face norm halved there: its natural
boundary condition is the vanishing viscous traction sigma_rr(r_w) = 0, and the
Dirichlet-closed gradient wall row (-2 p_0/h over the half cell, the closure a potential's
psi = 0 uses) gives the pressure force of the wall cell's value, the pair being the weak
form of the traction-free wall, sigma_rr - p = 0, of `Dt(u) - c_l Lap(u) + c_g Grad(p)
= 0` when c_l kappa = c_g with kappa = -c_p/c_T.  The residual traction (c_l kappa - c_g)
p_w is added on the wall faces over the half cell, -sigma (2/h)(c_l kappa - c_g) p_w with
p_w = (3 p_0 - p_1)/2 the second-order extrapolation from the two wall-most cells, so the
row is linear in the IR and any coefficient of p is honoured.  The flux-form divergence
reads the wall DOFs through the same closed pair, the tangential closure at the open wall
scales the theta derivative of v_r in the wall shear by the slip factor (1 at no-slip, 0 at
stress-free), and the multiplier's m = 0 gauge is lifted: the closed gradient wall row
breaks the constant pressure.  The open wall needs `Lap(u)` in u's evolution equation.
Verified on the traction-free Stokes ball against the poloidal roots of the 2 x 2 Bessel
determinant of tests/test_dedalus_ball.py at c = 1 and c = 2 (l = 2, c = 2: 8, 28.211083;
l = 1: 14.978747 for every c) and the toroidal family, at second order.

**Order > 2 at a Dirichlet wall.**  The flux stencils of order > 2 reflect the source nodes
with the odd parity about a Dirichlet wall (`_stag_diff_1d_wall_parity`), the treatment the
origin gets, so every row within the stencil window of the wall is a centred stencil on the
odd-extended field and the transposed divergence stays consistent through the wall cells;
a one-sided wall row alone leaves it inconsistent there (the composed operator then
converges at order 1).  The mirror ghost misreads the even part of the field about the
wall whatever the stencil order, so the wall cells converge at order 2 while the interior
keeps the stencil order; the same holds for the natural (Neumann) closure, whose wall row
is zero with the clamped near-wall rows, and for the pole and origin rows.  Reaching the
stencil order at the wall would need a staggered summation-by-parts pair whose boundary
rows and face weights are solved jointly, whose diagonal-norm boundary order is 2 in any
case.  The coverage cells of the scalar operators at order_r = 4 measure p = 2.0-2.1 on the
shell with errors 2-4x below order 2, p = 1.8-2.0 on the ball at m = 0, 2, and p = 1.0 at
m = 1 on the ball, where the origin ring's truncation at order 4 is O(1/h) with any wall
closure.

## 6. Ball geometry

Origin ring and pole closures as in the audited assembler (`_apply_ball_origin_vr`,
`_reduce_velocity_dofs`, `_origin_parity`); the flux stencils and the curl carry the origin
parity.  A pencil without multipliers and involutions (B = I, the compressible system) has
every equation's rows scaled by r^n, n = max(1, the natural r-power of its terms), the
origin ring at the half-cell radius, as the compressible hand assembler does; a
saddle-point or constrained-transport pencil is not scaled (the flux rows are incidence
sums with no 1 / r).

Constrained transport on the ball (docs/ball_staggered.md section 7).  The origin ring of
the face fluxes crosses faces of zero area, so it is identically zero with identically
zero tendency (the circulations on the origin's theta- and phi-arcs vanish by length) and
is pinned at every m -- a Dirichlet row with no mass beside an inviscid velocity, dropped
by the symmetric reduction (`_reduce_velocity_dofs(origin='drop')`) otherwise -- the way
the wall rings of a flux unknown are; there is no fold, since a flux through a zero-area
face carries no value to tie to a Cartesian vector.  Div Curl = 0 is combinatorial and
holds on the origin cell with no closure value (`_check_div_curl` passes to rounding at
every m), and regularity of the first ring of fluxes is the first cells' divergence, an
identity of the evolution, so no regularity rows exist for a flux.  The resistive blocks
are the shell's (every radial stencil is the centred two-point difference across a face
at r >= h).  The one closure is in the Lorentz force and the background-current block,
whose first-centre r-averages read the origin node (a zero-length edge, a zero-area
face, a guarded corner) as the parity reflection of the first interior node,
(3 + p) / 4 of it with p the vector origin parity (`CTGrid._origin_average`): the node
itself at m = 0 and |m| = 1, where j_theta(0), j_phi(0), b_r(0) are finite (the
|m| = 1 pole prolongation's form), half of it otherwise (the guard's zero average); the
plain average would read half the first-ring force at m = 0 and |m| = 1.  The induction
blocks need no origin value of v.  The vector potential on a ball is a regular vector and
takes the velocity's origin fold (m = 0, |m| = 1) or drop (|m| >= 2) with its wall rows
kept (`_reduce_velocity_dofs(walls=False)`); its folded master row is empty in A, since
the dual curl's origin ring rows are guarded and the origin face has no weight in the
gradient's adjoint, so it contributes one decoupled lambda = 0 eigenvalue as the inviscid
anchor does.  An involution unknown with no wall condition (the Malkus problem) keeps its
wall fluxes as DOFs evolving with the unmasked wall circulations (section 5, kind 'free').
The Lorentz force on the velocity's origin ring is zero (no flux crosses the origin
face), so the ring stays receive-only in the inviscid pencils and a viscous fold carries
the strain's ring rows and not the Lorentz force's, as it does not carry the advection's.

The origin closure of the vector operators (`docs/ball_staggered.md`, section 2).  The
origin ring v_r(0, theta_j) is one Cartesian vector: zero for |m| >= 2, one amplitude with
the profile p_j = cos(theta_j) (m = 0) or sin(theta_j) (|m| = 1) otherwise, onto which the
viscous reduction folds the ring, P^T A P with P the profile prolongation.  For the fold
to be an equation the ring rows must carry one: the strain Gram gives the ring the weight
(h^3/15) sin(theta_j) dtheta (`_origin_ring_volume`), so its rows are the first cell's
momentum balance and the folded master row is the profile-weighted average of the ring's
equations over the same average of its mass; the weight is the one that balances the
fold of the discrete Gram, -(4h^3/45) a_{div sigma} on a smooth solenoidal field, against
the folded mass (4/3) w_0.  The norm adjoints keep the zero weight (no flux crosses the
origin face), so the ring stays receive-only in the inviscid pencils.  Two origin-adjacent
stencils are
re-weighted on a ball because the midpoint flux forms are O(h^2 / r^k)-consistent, which
is O(1/h) on the first face or cell and answers with an error h^2 / r, first order in the
max norm: the strain spells r d_r(v_tan / r) as d_r v_tan - v_tan / r (the two-point
stencil exact on v_tan in {1, r}, where the two-point difference of v_tan / r is 33 % off
on the constant part of a field finite at the origin), and `build_staggered_grad_div`
differences a divergence whose radial flux form carries the face weights
r_{c,i-1} r_{c,i} = r_i^2 - h^2 / 4 (exact on v_r in {1, r} in every cell; the ring enters
the first cell as v_r(0) / h), giving up the exact self-adjointness of that block near the
origin.  The primal curl reads the pole v_theta through the |m| = 1 prolongation as the
strain does (section 3b).  With these every staggered ball cell of the coverage matrix
converges at order 1.8 to 2.2; the l = 1 toroidal decay rates lose the cubic exactness of
the old strain stencil (9e-5 to 2.6e-3 at 24 x 48, still second order).  The contraction
Dot(B0, Grad v) leaves the origin ring rows empty (section 2b); Curl(Cross(u, B0)) on a
plain face unknown compiles on a ball through the same CT blocks divided by the face
areas (the ring's circulations have zero length), and the coverage cell converges at
order 2; an involution unknown compiles through the same blocks with its origin ring
pinned.

## 7. What the new path replaced (stage 5)

`compile_fd2_staggered` is now one call: walk the system, classify the constraints,
dispatch the tasks, apply the BCs and the reductions, and return
`StaggeredCompiledMatrices` with a `constraints` record.  The `assembler=` keyword went
with the path it selected.

Deleted from `staggered_compile.py` (1155 lines, 2906 -> 1751): the recognisers
`_detect_velocity_pressure`, `_staggered_has_unsupported_terms` and
`_parse_incompressible_pair`; the two task readers that fed the hand assemblers,
`_compressible_spec_from_tasks` and `_mhd_spec_from_tasks`; the hand assemblers
`_assemble_inertial_cgrid` and `_assemble_compressible_cgrid`; the router
`_compile_fd2_staggered_pattern`; and the helpers left without a caller
(`IncompressibleSpec`, `_field_names_in`, `_real_if_real`, `_require_z_axis`,
`_tangential_wall_bc`, `_contains_type`, `_flatten`, `_scalar_factors`, `_peel`,
`_coriolis_partner`, `_coriolis_theta_block`, `_center_diag`, `_bg_scalar_eval`,
`_bg_vec_r_eval`, `_constant_value`).  Deleted from `staggered_mhd.py` (76 lines, 453 -> 377):
`build_mhd_blocks`, `extend_with_mhd` and their helpers `_merge` and `_scalar`.

What stayed: every mimetic block builder, because the dispatcher calls them -- the grids
and the layout, the 1-D stencils and their parities, the gradient blocks and the
norm-adjoint divergence pair, the curls and the curl-curl, the strain and the weighted
viscous Gram, the scalar Laplacians, the wall eliminations, the symmetric DOF reduction
and the ball origin rows; and in `staggered_mhd.py`, `CTGrid` with the four
`ct_*_blocks` builders, which is a coherent constrained-transport library and stays its
own module rather than moving into the operator file.

The acceptance identities, measured against the hand assemblers before the deletion and
recorded here because their reference no longer exists:

- inertial (`_assemble_inertial_cgrid`) on the Zhang system: max |A - A_ref| = 0, likewise
  B and the layout, on shell and ball, m = 0, 1, 2, inviscid and viscous with stress-free
  and no-slip walls, through the object API and through strings (stage 2);
- compressible (`_assemble_compressible_cgrid`) on the Bekki system entered as strings:
  the same layout, B exact, A exact at Ek = 0, and with Ek != 0 every block exact but the
  viscous (<= 8e-15 of its maximum, the density weight recovered by quadrature) and the
  thermal (2e-16) (stage 3);
- CT MHD (`extend_with_mhd`) on the dipole system: the same layout, B exactly and A
  exactly at Ek = 0; with Ek != 0 only the momentum row's viscous and Coriolis blocks
  carry that rounding (<= 7.8e-15 of the block maximum) and every magnetic block is exact
  (stage 4);
- the object system of `make_polytrope_system`, respelled `Grad(c q)` in stage 5: the
  hand pencil from the respelled system is bitwise the hand pencil from the product-rule
  spelling (the hand assembler recombined the pair by name), and the general pencil then
  agrees with it to 9.3e-15 absolute, 7.7e-22 of the global maximum, with the four
  pressure blocks exactly equal and the whole residue in the viscous and Coriolis blocks
  at <= 3.3e-16 of their own maxima.

The tests that read a hand pencil were rewritten against invariants of the general path
rather than deleted: the two spellings of a system (IR objects and `Problem` strings) are
required to give one pencil entry for entry, the flux rows of a CT pencil are required to
be annihilated by the flux divergence, the ball pencil's row premultiply is read off B,
and the classification record replaces what the recognisers used to return.

## 8. New verification beyond the existing systems

- Boussinesq convection in a shell: onset Rayleigh number and frequency against the Dedalus
  core on the same strings (no closed form), and the non-rotating spherical benchmark of
  Chandrasekhar for internal heating with stress-free walls where the critical values are
  tabulated.
- Anelastic: the Bekki polytrope with `Div(rho0 v) = 0` in place of the continuity
  equation, against the Dedalus core.
- Induction with the b formulation and the involution: free-decay modes of a conducting
  sphere/shell (toroidal and poloidal decay rates from spherical Bessel conditions), Alfven
  modes on a uniform axial field, against the Dedalus core and the closed forms.
- A formulation with the Coulomb gauge as a saddle-point constraint on A, against the b
  formulation.
- Each refusal path: one test per unrecognised constraint shape and BC kind, asserting the
  message.

Delivered (stage 4).  **Free decay in a conducting shell** (r_i = 0.5, r_o = 1, eta = 1,
m = 0, `tests/test_staggered_general.py::TestFreeDecay`).  With
b = Curl(Curl(S r)) + Curl(T r) both potentials obey Dt(X) = Lap(X), so
X = a j_l(k r) + b y_l(k r) and the rate is -k^2; b_r = l(l+1) S / r gives S = 0 at the
walls, which also kills the toroidal part of E = Curl(b) (proportional to Lap(S) = -k^2 S),
and the poloidal part of E is proportional to d(r T)/dr, so the toroidal potential closes
on d(r T)/dr = 0.  The determinants are therefore
`j_l(k r_i) y_l(k r_o) - j_l(k r_o) y_l(k r_i) = 0` (poloidal) and the same with
`f -> d(r f(k r))/dr` (toroidal).  Relative errors of the flux (b) formulation:

| family | l | n | rate k^2 | 12x24 | 24x48 | 48x96 | order |
|---|---|---|---|---|---|---|---|
| toroidal | 1 | 1 | 3.938519 | 3.82e-3 | 9.78e-4 | 2.46e-4 | 1.97, 1.99 |
| toroidal | 1 | 2 | 43.804079 | 4.85e-3 | 1.21e-3 | 3.03e-4 | 2.00, 2.00 |
| toroidal | 2 | 1 | 11.464154 | 1.18e-3 | 3.70e-4 | 9.84e-5 | 1.68, 1.91 |
| poloidal | 1 | 1 | 43.191357 | 5.52e-3 | 1.38e-3 | 3.46e-4 | 2.00, 2.00 |
| poloidal | 2 | 1 | 50.574517 | 6.51e-3 | 1.63e-3 | 4.08e-4 | 2.00, 2.00 |

and of the vector potential (A): toroidal l = 1 n = 1 1.73e-4 / 4.75e-5 / 1.21e-5
(1.86, 1.97), toroidal l = 1 n = 2 6.42e-3 / 1.61e-3 / 4.02e-4, toroidal l = 2 n = 1
6.07e-3 / 1.51e-3 / 3.79e-4, poloidal l = 1 n = 1 6.20e-3 / 1.55e-3 / 3.89e-4, poloidal
l = 2 n = 1 4.94e-3 / 1.22e-3 / 3.05e-4, every one at order 2.  The toroidal family needs
the m = 0 pole closure of j_r (`CTGrid(pole_closure=True)`, the general path's default):
without it the toroidal rates stall at order 0.2 (0.42, 0.35, 0.31 for l = 1 n = 1),
while the poloidal family, which does not see j_r at the pole, is unaffected.  For
|m| >= 1 regularity makes j_r vanish at the pole and the closure is inert (the two
`jr_bphi` matrices are bitwise equal), so the pattern path's `pole_closure=False` and the
general path agree wherever the certification runs.

**b against A on the same physics** (2 kG dipole on the Bekki polytrope, conducting,
m = 8, `tests/test_staggered_mhd.py::test_b_and_A_formulations_agree_on_the_field_induced_shift`).
Field-induced Rossby shift, general path: 10x24 db = -0.06439-0.00738i,
dA = -0.06849-0.00963i (|db - dA| = 4.68e-3); 20x48 db = -0.06757-0.00901i,
dA = -0.06853-0.01027i (1.58e-3).  The flux formulation lands on the seed Dedalus value
-0.0676-0.0091i and the vector potential on the spdee Dedalus value -0.0686-0.0102i, the
two references of `test_shift_matches_dedalus_backend`; the two staggered formulations
close on each other at about first order between these two grids (the difference is the
wall layer each resolves, not the interior stencil, which is second order in both).  Both
are div-free: the flux formulation to 1e-13 of the flux maximum through
`StaggeredCompiledMatrices.divergence`, the vector potential exactly (its Coulomb gauge
holds to 4e-16 and Div Curl A = 0 to 7e-16 on the dual complex).

**Ideal Alfven modes on a uniform axial field** (r_i = 0.5, r_o = 1, |B0| = 1, m = 1,
`TestIdealAlfven`).  The ideal spectrum is the Alfven continuum of the field lines, whose
gravest frequency is pi / (2 sqrt(r_o^2 - r_i^2)) = 1.81380 (the longest line, grazing the
inner sphere at the cylindrical radius s = r_i); single eigenvalues therefore do not
converge.  The C-grid band closes on that edge from below (lowest |omega| 1.0416 at 12x24,
1.4884 at 24x48) and the Dedalus core from above (2.0884 at 16x24, 1.9417 at 32x48).  With
the ideal wall Dot(t_hat, Cross(v, B0)) = 0 read as the tangential electric field of the
flux unknown (so its wall circulations are masked) the C-grid spectrum is non-growing:
max |Im omega| = 7.4e-14 at 12x24 and 6.7e-3 at 24x48 against 2.8e-14 / 1.3e-13 for
Dedalus.  Read instead as a condition on the velocity -- which is what the walker's
`unknown_name` says, there being no Curl in it -- the wall circulations are left alive, the
flux tendency stops being a pure curl at the wall rows, and both the involution and the
non-growth are lost (max |Im omega| 0.58 at 12x24, and eigenvectors with
|Div Phi| > |Phi|).

Not delivered: the norm-adjoint ("Hodge") pairing of the CT Lorentz force.  Building the
four edge-to-face maps of `ct_lorentz_blocks` as W_face^-1 P^T W_edge of the induction
interpolations, with B0 at the edge, does not make the ideal pencil skew (max |Im omega|
0.34 at 12x24, 0.17 at 24x48, no better than the CT recipe's 7.4e-14 / 6.7e-3 once the
wall mask is right) and it breaks the entrywise identity with the certified code, so the
plain two-point averages of the seed's recipe are the only Lorentz pairing.

## 9. Staging

1. Dispatcher skeleton and the block library audit: every task kind either maps to an
   existing builder or is listed as missing (this document's table, kept current).
2. Saddle-point path: inertial and viscous rotating flow, Boussinesq, anelastic; the
   entrywise identity with `_assemble_inertial_cgrid`.

   Delivered (stages 1 and 2): `spdee/backends/fd2/staggered_general.py` with
   `classify_constraints` (section 4: evolution / saddle point with the multiplier, the
   Div weighting M, the divergence sign and outer coefficient, the m = 0 gauge rule; the
   involution is classified and then refused until stage 4; tensor unknowns refused with a
   pointer to section 2b; every other algebraic shape refused with the equation, the shape
   found and the two supported shapes), `compile_staggered_general` (the dispatcher of the
   table above, the wall record of section 5, the ball origin of section 6, returning
   `StaggeredCompiledMatrices` with `constraints`, `layout.placements` and `b_total`), and
   `general_path_reason` / `general_path_applies`.  `compile_fd2_staggered(assembler=)`
   routes: 'auto' (default) takes the general path when every equation is an evolution
   equation or a saddle-point constraint with at least one multiplier and all kinds are
   stage-2 kinds, else the pattern path (compressible, CT MHD, the Dt-less Zhang form) with
   the classifier's reason appended to a pattern refusal; 'general' and 'pattern' force a
   path.  `Problem.compile(discretization='staggered')` therefore reaches the general path
   for every saddle-point system.  The walker gained two backend-agnostic expansions:
   `Div(M u)` by the product rule into a tagged task pair, and background scalars folded
   from either side of `s * B0` (so `Ra*T*r_hat` walks).  Acceptance
   (`tests/test_staggered_general.py`): the entrywise identity with
   `_assemble_inertial_cgrid` is exact (max |A - A_inertial| = 0, likewise B and the
   layout) on shell and ball, m = 0, 1, 2, inviscid and viscous with stress-free and
   no-slip walls, through the object API and through strings; Boussinesq convection
   classifies as a saddle point with exactly W-skew Coriolis blocks (2e-16) and an exactly
   W-adjoint buoyancy / advection pair (1e-16) on the reduced pencil, no row empty in both A
   and B, and its least-damped eigenvalue at Ra = 2.5, Ek = 1e-2, m = 2 converges at second
   order (errors 1.12e-3, 2.77e-4, 6.89e-5 at 12x24, 24x48, 48x96) to the Dedalus value
   -0.02510814 + 0.04842610i on the same strings; the anelastic Bekki polytrope with
   `div(rho0*v) = 0` classifies with weight rho0 and its Grad / Div pair is adjoint under
   the rho0-weighted norm to 2e-16.  The pattern assemblers and recognisers stay in place
   until stage 5.
3. Compressible path through the dispatcher; entrywise identity with
   `_assemble_compressible_cgrid` on the Bekki system; delete the recogniser.  Tensor
   placement (section 2b): the strain-based terms as six-component tensors, tensor unknowns
   with the first-order reduction `T - Grad(v) = 0`, Div(T) and Dot(B0, T) blocks.

   Delivered (stage 3).  The walker distributes Grad over sums of background-weighted
   unknown scalars into tagged product-rule pairs (`grad(Cs2*(rhat+S))` walks), folds
   background scalars around `Div(v) * B0`, and walks tensor equations (identity_tensor,
   tensor_of_vector as one symbolic sum, source_tensor) and Div / Dot of an unknown tensor.
   `staggered_general` dispatches every kind but the three curl kinds through the table
   above, with the combinations read from the task list (the weighted gradient, the
   viscous force as one strain Gram with the density weight recovered from d ln rho0 / dr,
   the P0-weighted thermal Laplacian), the tensor placements of section 2b, the definition
   elimination, the ball premultiply of section 6, and `general_path_reason` accepting
   every system whose equations classify (the Dt-less Zhang form and the curl kinds still
   go to the pattern path).  Acceptance (`tests/test_staggered_general.py`):
   `solar_hydro_problem()` as strings, compiled with no recogniser, is the hand
   assembler's pencil (the momentum and energy rows of a rho0-weighted call, the
   continuity row of a unit-density call, since the strings carry rhat = rho1 / rho0)
   at 12x48 and 8x16, m = 8, 0, 1, Ek != 0 and Ek = 0, delta = 0, 'center' and 'seed':
   the same layout, B exact, A exact when Ek = 0, and with Ek != 0 every block exact
   except the viscous blocks (<= 8e-15 of their maximum, the density weight recovered by
   quadrature where the hand assembler reads rho0 itself) and the thermal block (2e-16,
   the stratification coefficient arrives folded with Ekap so its ratio to the Lap
   coefficient carries one rounding); SymPy constants (Ek, the Coriolis 2) are read
   exactly rather than through lambdify's 15 digits; the regular compressible ball (the
   builder of `tests/test_ball_staggered.py` with the pressure gradient spelled Grad(c q)) matches
   the pattern path exactly when Ek = 0, origin ring and premultiply included, and to
   1.2e-15 per block when Ek != 0.  The tensor placement: Grad then Div is the strain
   Gram with C = 2I to rounding, Transpose is a permutation, Trace of Grad and Dot(r_hat,
   Grad v + Grad v^T) agree with the divergence and the hand gradlnrho blocks to second
   order (ratios 3.7-4.0 over 12, 24, 48).  The tensor-unknown system `dt(v) - Ek div(T)
   + grad(p) = 0, T - grad(v) - trans(grad(v)) = 0, div(v) = 0` eliminates to the
   `div(grad(v) + trans(grad(v)))` pencil entrywise with the same spectrum to 2e-15; with
   Dt(T) the tensor keeps its six blocks.  The strain-divergence form of a Boussinesq
   system differs from `lap(v) + grad(div(v))` by exactly (2/3) Ek (G Div - K_tr), K_tr
   the trace Gram, whose first part vanishes on the discrete constraint manifold and whose
   second is O(h^2) there; the spectra agree at 1.8e-6 (12x12) and 1.1e-7 (24x24).
   Deleted from the pattern path: `_classify_compressible_scalars`, `_extract_ek`,
   `_extract_coriolis_coeff`, `_entropy_wall_bc`, `_staggered_has_grad_ln_rho_strain`,
   `_vector_unknown_name`, `_scalar_unknown_for_equation`, `_dt_target_names`;
   `_assemble_compressible_cgrid` stays reachable through `assembler='pattern'` (and
   through the constrained-transport route until stage 4) with its inputs read from the
   task list (`_compressible_spec_from_tasks`).  Not identical, deliberately: the object
   system of `make_polytrope_system` spells the pressure gradient by the product rule
   with numeric backgrounds, which the general path assembles as two blocks (c Grad q on
   the faces plus q Grad c), while the seed and the hand assembler use G(c q); the
   certification `scripts/certify_staggered_vs_seed.py` therefore pins
   `--assembler pattern` (the same (c) table as before; `--assembler general` changes only
   the vr,rho and vr,s rows, to 4.0e-2 and 8.1e-4), and `tests/test_staggered_mhd.py`
   builds its hydro reference on the pattern path too.  Stage 5 moves the certification
   to the strings (rhat, S variables with a per-row conversion) when the pattern path goes.
4. Involution path: CT induction through the dispatcher; entrywise identity with
   `extend_with_mhd`; the vector-potential formulation as a saddle-point case (A on the
   faces, psi at the centres, b = Curl A on the edges, Lorentz force through Curl Curl A),
   cross-checked against the b formulation spectrum by spectrum.

   Delivered (stage 4).  The walker gained `Curl(Curl(v))` and
   `Cross(Curl(Curl(v)), B0)` (`TASK_CURL_CURL_VECTOR`, `TASK_CROSS_CURL_CURL_UNK_BG`)
   and the boundary kind `BC_PROJECTION_CURL_CROSS`: `Dot(u_hat, X)` with X a
   constant-coefficient combination of `Curl(v)` and `Cross(v, B0)` terms, the tangential
   electric field of a conducting wall.  `classify_constraints` classifies the involution
   of section 4 case 2 -- written (`Div(b) = 0` with no multiplier, its row dropped,
   `equations[eq]['class'] = 'involution'`) or implied (an evolution whose A-terms are
   only curls of the state, which is what `add_magnetic_field` produces) -- records
   `involutions[b]` and `constraints[b] = 'involution'`, and refuses a constraint that
   admits both readings, naming the multiplier and the curl terms.  The dispatcher routes
   `curl_cross_unk_bg` to `ct_induction_blocks`, `Lap(b)` / `Curl(Curl(b))` on an
   involution to `ct_resistive_blocks` (the rewrite of section 4: `Lap(b) = -Curl Curl b`
   on the manifold, plus `Grad Div b`, which vanishes there) and `cross_curl_unk_bg` to
   `ct_lorentz_blocks`, all through `_CGridOps.ct()`; `_check_div_curl` asserts
   `||Div A_b|| = 0` to 1e-14 of the block maximum at assembly and
   `StaggeredCompiledMatrices.divergence(x)` returns the net flux out of each cell of the
   reduced state.  The vector potential is the saddle point of the same face placement:
   `Div(A) = 0` with psi at the centres, `Lap(A) = -Curl Curl A` through
   `_CGridOps.curl_curl` (`potential_diffusion`), the Lorentz force
   `Cross(Curl(Curl(A)), B0)` through the Hodge-paired cross block, tangential `A = 0`
   closing the curl stencils (which is `b_r = 0` at the wall) and `psi = 0` closing the
   gradient wall rows with the half-cell face weight, so the flux-form divergence stays
   the norm adjoint of the gradient and the wall `A_r` DOFs are kept.  With psi Dirichlet
   at both walls the constant multiplier is no longer a null vector, so the m = 0 gauge is
   dropped; `integ(psi) = 0` is still applied when the walls leave psi free.  Wall
   conditions come from the BC tasks alone: `b_r = 0` at every wall (radial) with, for a
   conducting wall, the tangential electric field on both `theta_hat` and `phi_hat`,
   checked in `_check_involution_walls` to be the field the evolution carries; a
   tangential electric field that names a non-flux unknown (the ideal
   `Dot(t_hat, Cross(v, B0)) = 0`, which has no Curl in it) is re-attributed to the
   involution unknown by `_electric_target`; pinned tangential components, a mixed
   closure, a radial electric field and an involution on a ball are refused.
   Acceptance (`tests/test_staggered_general.py`): the dipole system through the
   dispatcher is `extend_with_mhd`'s pencil, the same layout and B exactly and A exactly
   at Ek = 0, at 12x48 and 8x16, m = 8 and m = 1, conducting and radial walls, with Ek != 0
   only the momentum row's viscous and Coriolis blocks carrying the stage-3 rounding of
   the quadrature-recovered density weight (<= 7.8e-15 of the block maximum, absolute
   <= 4.3e-16) and every magnetic block exact; with the pressure gradient spelled by the
   product rule (the object system of `make_polytrope_system`) the deviation is
   bitwise the field-free one, so constrained transport adds nothing to it.  `'auto'` now
   routes a constrained-transport system to the general path;
   `build_mhd_blocks` / `extend_with_mhd` stay reachable through `assembler='pattern'`,
   which is what `scripts/certify_staggered_vs_seed.py` and `tests/test_staggered_mhd.py`
   pin, and `recognize_mhd` / `MHDSpec` are deleted (the pattern path reads the same
   inputs from the task list through `staggered_compile._mhd_spec_from_tasks`).  Section 8
   carries the closed forms.
5. Delete the hand assemblers; update the README and the operator coverage matrix.

   Delivered (stage 5).  The pattern path is gone, with the `assembler=` keyword;
   `compile_fd2_staggered` is one call into `compile_staggered_general`, and section 7
   lists what was deleted, what stayed and what the acceptance identities were.

   Three things had to change before the deletion could be entrywise-clean.
   (a) **The pressure gradient of the object system.**
   `make_polytrope_system` and `make_solar_ball_system` spelled it by the product
   rule with numeric background gradients; the hand assembler recombined that pair by
   name into `G diag(c)`, so the certified operator was the grouped one all along, and the
   general path -- which builds what is written -- differed from the seed in the
   `(v_r, rho1)` and `(v_r, s1)` blocks by 4.0e-2 and 8.1e-4 relative.  Both builders now
   spell it `Grad(c q)`, the seed's and `solar_hydro_problem`'s spelling.  Measured at
   12x48, m = 8, nu = kappa = 2e12, delta = 1e-6: the hand pencil is bitwise unchanged by
   the respelling (max |A_new - A_old| = 0 over 2844 x 2844), the general pencil then
   agrees with it to 9.3e-15 absolute (7.7e-22 of the global maximum) with all four
   pressure blocks exactly equal and the residue confined to the viscous and Coriolis
   blocks at <= 3.3e-16 of their own maxima, and both reproduce the seed's `vr,rho` and
   `vr,s` rows at 5.6e-16 and 1.7e-15.
   (b) **Numeric weights under Grad and Div.**  `_emit_weighted_grad` required a
   SymPy-backed weight so it could form `Grad(M)` for the collocated product-rule pair;
   the polytrope's backgrounds are numeric.  The walker now synthesizes that gradient by
   central differences of step 1e-4 for a numeric weight and by `sp.diff` for a SymPy one
   (`_background_gradient_field`, shared with `_emit_weighted_div`), which is the rule
   `spdee.ir.backgrounds` already states for derivatives of numeric components.  The
   staggered path never evaluates it: it consumes the tagged pair as one `G diag(M)`.
   (c) **The Dt-less spelling.**  `2 Cross(Omega, v) - Grad(p) = lambda v` (the Zhang
   form, used by `tests/test_evp_zhang_2001.py`, `tests/test_ball_staggered.py` and
   `scripts/inviscid_order_sweep.py`) was the last system the pattern path still owned.
   Section 4 now reads a system with no `Dt` anywhere as the eigenvalue operator itself
   and gives its vector unknown the identity in B, which is what the hand assembler did
   with `a = 1`; a system that mixes `Dt` and `Dt`-less equations is unaffected.  The two
   spellings compile to the same pencil, A and B both exact.

   The certification (`scripts/certify_staggered_vs_seed.py --nr 12 --nt 48`) now runs the
   general path; the `ASSEMBLER` pin, its comment and the `--assembler` flag are gone.  Its
   156-line report is line for line the pinned pattern-path one -- the (c) block table, the
   per-term alpha table, the residual-after-alignment table, the theta profiles, the
   attribution tables, the south-pole closure, the hybrid Rossby modes and the (b) dense
   spectra -- except three entries at the noise floor: two alpha imaginary parts
   (`viscous vr,vp` -2.4e-16 -> -1.4e-16, `viscous vp,vt` -0.0 -> -5.3e-17, both against a
   real part of 0.9997) and one hybrid Rossby difference (-5.55e-13-1.06e-12j ->
   -3.75e-13-8.80e-13j).  `tests/test_staggered_mhd.py` and every other test that pinned
   `assembler='pattern'` were unpinned.

   Two numbers in the stage-4 record above moved with the respelling, both inside the
   tolerances already asserted.  The field-induced Rossby shift of the flux formulation is
   db = -0.06436-0.00749i at 10x24 and -0.06757-0.00904i at 20x48 (was -0.06439-0.00738i
   and -0.06757-0.00901i), the vector potential's dA = -0.06847-0.00973i and
   -0.06853-0.01029i, so |db - dA| is still 4.68e-3 and 1.58e-3 and each formulation still
   lands on its own Dedalus reference.  Reaching them needs the ARPACK shift moved 0.01
   below the mode in Im(omega) (`tests/test_staggered_mhd.py::rossby`): with the shift
   sitting on the Rossby mode itself the 20x48 flux pencil stalls at 5 of 8 converged
   vectors, a conditioning accident of the new pencil rather than a change in the mode --
   the damped shift returns the same eigenvalue on every case that converged either way.

6. The vector Laplacian off the constraint manifold, the pole rings, and what the Zhang
   inertial-mode tests measure.

   Delivered (stage 6), the three defects the coverage matrix and the closed-form
   benchmarks found.

   (a) **`lap(v)` built a different operator from the one written** unless v was
   divergence-free (section 3a).  `_Assembly.viscous_force` now decides from
   `record['constraints']` and from the group's own `Grad(Div(u))` coefficient.  The
   certified identities are unchanged: `scripts/certify_staggered_vs_seed.py --nr 12
   --nt 48` is line for line the pinned report except the two (a) lines the pin moves
   (nnz 28676 -> 28700 and max |K| on the pole rows 0 -> 1) and the three noise-floor
   entries the stage-5 record already names, and the inertial saddle-point and Bekki
   compressible pencils are bitwise what they were off the pole rows.  In the coverage
   matrix the staggered core goes from 36 PASS / 90 FAIL to 90 PASS / 36 FAIL, and every
   remaining failure is on the ball origin or the |m| = 1 pole of the curl.

   (b) **The pole rings were pinned by accident** (section 3b).  On the inertial ball
   pencil at 16x32 the dense census loses exactly 32 = 2 N_r eigenvalues at
   |lambda| = 5.6e-17 and every other eigenvalue is bitwise unchanged (m = 1 and m = 2,
   976 residual-gated modes either way); on the Bekki compressible pencil at 12x48 the
   certification's dense window is unchanged (611 modes in window, Rossby
   -0.21982-0.01913j, max |d omega| 2.641e-01 against the seed, identical before and
   after) because "spdee as solved" was already the south-pole-dropped operator there.
   `A_total` and `B_total` are bitwise unchanged off the pole rows in both.  The
   inertial-mode census of `tests/test_analytic_evp.py` sees the same 2 N_r drop:
   288 neutral modes in the band at 10x16 and 1212 at 20x32 against 308 and 1252,
   with no target or control distance moved (docs/analytic_benchmarks.md).

   (c) **The staggered Zhang tests were not evidence.**  The inviscid band is dense (960
   neutral modes at 16x32, 2206 at 24x48), so the 1.5e-2 / 6e-3 per-target thresholds sat
   at or below the mean spacing.  `tests/test_ball_staggered.py::TestZhangBall` now
   asserts the RMS error over the ten targets (2.20e-3 -> 7.72e-4 at m = 1, order 2.58;
   3.12e-3 -> 7.11e-4 at m = 2, order 3.65) and reports the discrimination ratio, which
   this core does not earn (0.83 and 1.64 at m = 1, 1.00 and 1.08 at m = 2, against
   5.5-13.6 for the collocated core and ~3300 for sh_fd and Dedalus).
   `tests/test_evp_zhang_2001.py::TestZhang2001StaggeredInertialWaves` keeps only the
   claim its measurement supports, that a mode lands within 0.03 of every target across
   the full band (the guard against a half-strength Coriolis, which truncates the band at
   |omega| ~ 1.1), and runs one census per m instead of one per target.
   `TestCompressibleBall::test_inertial_modes_match_incompressible_assembler` does
   discriminate -- the matched modes agree to 5e-5 while a random control frequency lands
   2.3e-3 away, a ratio of 540 to 830 -- so its threshold was tightened to 1e-4 and the
   ratio is asserted above 100.

7. The remaining refusals and pole limits of the coverage matrix.

   Delivered (stage 7).  (a) **A theta-dependent background in Dot(B0, Grad(u)), Dot(B0,
   tensor(u)) and Dot(B0, T)** through `_CGridOps.dot_vector` and the paired spelling of
   the nine-component map (section 2b): the eighteen Dot cells of the staggered column
   converge at order 1.95 to 2.07 on shell and ball, m = 0, 1, 2, and the Doppler shift of
   the inertial modes in a rotating frame (`tests/test_analytic_evp.py`) is exact to
   3e-15 at 10x16 and 20x32 while the frequencies land within 4.6e-3 of Zhang et al.;
   `tests/test_staggered_general.py::TestBackgroundAdvection` asserts the entrywise
   identity Dot(s phi_hat, Grad v) = i m v + Cross(z_hat, v) against the assembled
   Coriolis block on shell and ball.  The certified Bekki pencils are untouched: the
   strain contraction `dlnrho * Dot(er_, Grad v + Grad v^T)` is consumed by the
   density-weighted Gram before any contraction is built.  (b) **Div(T) of a
   non-symmetric tensor unknown** converges at order 2 at m = 0, 1, 2 on shell and ball
   (the pole rows, the pole share and the wall closure of the placement spelling); the
   six-component path is entrywise what it was.  (c) **The |m| = 2 pole-adjacent v_phi
   ring** is diagnosed in section 3b and closed behind `pole_closure=`, off by default.
   (d) **Curl(Cross(u, B0)) under a no-slip u** compiles as a plain face unknown (section
   4), and the cell converges at order 2 on shell and ball; constrained transport proper
   on a ball stays refused, the last staggered gap.  The staggered column of
   `docs/OPERATOR_COVERAGE.md` goes from 132 PASS / 24 REFUSED to 156 PASS.

8. Constrained transport on the ball.

   Delivered (stage 8), the last staggered gap, by the origin closure of section 6 and
   docs/ball_staggered.md section 7.  The refusal `INVOLUTION_ON_BALL` and the potential's
   ball refusal are gone; `_apply_gauge_and_walls` pins the origin ring of every flux
   unknown at every m (Dirichlet beside an inviscid lead, dropped by
   `_reduce_velocity_dofs(origin='drop')` beside a viscous one), folds or drops a
   potential's origin ring with its wall rows kept, and keeps the wall fluxes of a free
   wall; `CTGrid._origin_average` closes the first-centre r-averages of the Lorentz and
   background-current blocks by the vector origin parity; `_classify_walls` gives an
   involution unknown with no wall condition the kind 'free', which
   `_check_involution_walls` accepts on an ideal evolution only.  Acceptance
   (`tests/test_staggered_general.py`, `tests/test_ball_staggered.py`,
   `tests/test_analytic_evp.py`): `_check_div_curl` passes on the ball at m = 0, 1, 2, 8
   with the origin ring rows empty; the Lorentz force of a uniform current (b = j x r / 2)
   in a uniform axial field (|m| = 1) or a radial background (m = 0) is exact on the first
   ring where the interior is, against half of it with the guarded average; free decay in
   a perfectly conducting ball against the Bessel conditions (toroidal (x j_l(x))' = 0,
   poloidal j_l = 0, l = max(m, 1)) at m = 0, 1, 2 over 12x24 and 24x48 converges at order
   1.88 to 2.10 in the flux formulation (errors 1.6e-4 to 3.8e-3 at 24x48) and 1.98 to
   2.10 in the vector-potential formulation (2.7e-4 to 3.2e-3), the two agreeing to their
   discretization errors; the Malkus benchmark at m = 2 runs on the staggered core with
   the free wall (errors up to 2.8e-3 at 20x32 against the frequency unit, RMS order 2.2
   to 2.3, the product relation to 1e-6; tolerance 5e-3); the `Curl(Cross(vector, B0))`
   and `Cross(Curl(vector), B0)` ball cells of the coverage matrix are PASS at order
   1.98 to 2.09, unchanged.  The shell is bitwise what it was: the constrained-transport
   pencils of `TestConstrainedTransport`, the shell decay and Alfven pencils, and the
   certification `scripts/certify_staggered_vs_seed.py --nr 12 --nt 48` is line for line
   the pinned report except four dense-census entries at the 1e-14 noise floor and a
   timing.
