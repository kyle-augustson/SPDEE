# The ball (r_inner = 0) on the staggered Arakawa-C core

Audit of the origin treatment in `spdee/backends/fd2/staggered_compile.py`,
the defects fixed, and the analytic benchmarks of `tests/test_ball_staggered.py`.
Resolutions are radial cells × θ cells (`make_ball_grid(N_r=cells + 1, ...)`).

## 1. What lives at the origin

The origin is the inner r-edge `r_edge[0] = 0`. Only `v_r` (and, in MHD,
the flux `Φ_r`) lives on r-edges, so the origin carries the ring
`v_r(0, θ_j)`, j = 0 … Nθ−1: Nθ copies of one physical point. `v_θ`, `v_φ`,
`p`, `ρ₁`, `s₁` sit on r-centers strictly inside (`r_center[0] = h/2`).

Analytic behaviour of a Cartesian-regular field with e^{imφ} dependence:

| field | at r = 0 | m-content of the origin value |
|---|---|---|
| vector `v(0) = c` | `v_r(0, θ) = c·r̂(θ)` | `\|m\| ≥ 2`: 0; `m = 0`: `c_z cos θ`; `\|m\| = 1`: `c_⊥ sin θ` |
| scalar `f` | `f ~ r^{\|m\|}` | `m = 0`: finite; `m ≠ 0`: 0 |

Along a ray at fixed θ a regular field extends to negative r with the
parity of its lowest Cartesian monomial: `f(−r) = (−1)^{\|m\|} f(r)` for
scalars and `(−1)^{max(\|m\|−1, 0)} f(r)` for every vector component
(`_origin_parity`). This is exact for the equatorially symmetric
family (l ≡ |m| mod 2) and is the closure used by every radial stencil that
touches the origin row (`_stag_diff_1d_origin_parity`): the gradient
`G_r` of the pressure-like scalars, the flux-form scalar Laplacians, and the
radial derivatives of `r·v_tan` (curl) and `v_tan/r` (strain), the last two
with the parity of the divided/multiplied field (`−(vector parity)`).

## 2. How the assembler closes the origin

* **Zero area, zero weight.** The origin face has area 0 and the volume
  weight of the origin ring `W_vr[0] = r_edge[0]² sinθ Δr Δθ = 0`. Every
  norm-adjoint operator (`Div = −W_p⁻¹ Gᵀ W_v`, the Coriolis partner
  `P_vr2vp`, the buoyancy/entropy pair `P_vr2c`, the viscous Gram rows)
  therefore ignores `v_r(0, θ)`: no flux crosses the origin, the first
  cell's divergence uses only its outer face. In the inviscid pencils the
  origin ring is *receive-only* (`test_origin_rows_and_receive_only_ring`);
  in the viscous pencils it enters through the strain (`e_rr = ∂_r v_r`
  at the first center, `v_r/r` interpolations), which is why the viscous
  path reduces it symmetrically.
* **Regularity rows (`_apply_ball_origin_vr`, inviscid).** `|m| ≥ 2`:
  Dirichlet-zero rows (A = 1, B = 0 → infinite eigenvalues, filtered).
  `m = 0` / `|m| = 1`: rows j > 0 become
  `v_r(0, θ_j) − (p_j / p_0) v_r(0, θ_0) = 0` with `p = cos θ` / `sin θ`,
  no mass; the j = 0 anchor keeps its assembled momentum row. Because the
  ring is receive-only, that anchor row has no self-coefficient and
  contributes one decoupled `λ = 0` eigenvalue.
* **Symmetric reduction (`_reduce_velocity_dofs`, viscous).** The
  prolongation P drops the outer-wall `v_r` ring, drops the origin ring for
  `|m| ≥ 2`, and for `m = 0`, `|m| = 1` folds it onto the (ir = 0, j = 0)
  master with the profile ratios `p_j / p_0`; `A_red = Pᴴ A P`,
  `B_red = Pᴴ B P` is a Galerkin projection of the Nθ origin rows onto
  the single Cartesian DOF and keeps the viscous operator self-adjoint.
  The master row is the profile-weighted average of the ring's momentum
  equations, `Σ_j p_j (A P)_j / Σ_j p_j²`, and its mass the same average
  of the ring's mass (the compressible premultiply `r_c[0]²` is uniform on
  the ring and cancels).
* **The ring's viscous row (`_origin_ring_volume`).** In the strain Gram
  `F = −W_v⁻¹ Sᵀ W_T C S` the origin ring takes the weight
  `W_ring = (h³/15) sinθ_j Δθ` instead of the guard's zero, so its rows
  are the momentum balance of the first cell. The ring basis function
  strains that cell only, `e = (−1, 1, 1) p_j / h` on the diagonal, so
  with the trace-free stress the fold is
  `Σ_j p_j (K u)_j = −(h²/2) Δθ Σ_j sinθ_j p_j σ_rr(h/2, θ_j)`; the origin
  value of `σ_rr` has no l = 1 content and its radial derivative
  integrates to `(8/15) a_{∇·σ}` against `sin²θ` (with `a_g` the origin
  amplitude of a vector, `ĝ_r(0, θ) = a_g sinθ`), so the fold of a smooth
  field is `−(2h³/15) a_{∇·σ}` and the mass `Σ_j W_j p_j² = (4/3) w₀`
  would balance it at `w₀ = h³/10` for the exact strain; the discrete
  strain adds the first-cell error of `v_r/r` (the two-point average
  over `r_c = h/2`, `(h/4) ∂_r² v_r` in `e_θθ`, `e_φφ`), whose fold is
  `(2h³/45) a_{∇²u + 2∇(∇·u)}`, so on a solenoidal field the fold is
  `−(4h³/45) a_{∇·σ}` and the weight is `w₀ = h³/15`. Measured on the
  manufactured `|m| = 1` field the effective weight is `0.063 h³`
  (`∇·u ≠ 0`, and the two-point θ stencil differentiates `cosθ` with a
  `sinc(Δθ/2)` factor, an `O(Δθ²/h) a` reaction term in the first cell's
  `e_θθ`); the master's own error is O(h³) either way, its value being
  fixed by the neighbouring rows, so the weight sets the ring's inertia
  and not the convergence. The two-point average of `v_r` over the first
  cell is kept: it is the weight that makes the transposed form,
  `−(σ_θθ + σ_φφ)/r` on the first faces, exact on a constant stress (a
  quadratic through the ring and the two faces beyond it makes `e_θθ`
  O(h²) there but that form 12 % off, O(1/h) on the `v_r` rows). The norm
  adjoints (Div, the Coriolis partners, the Hodge maps) keep the zero
  weight: no flux crosses the origin face.
* **Origin-adjacent stencils.** The midpoint flux forms of the C-grid are
  `O(h²/r^k)`-consistent, which is O(1/h) on the first face or cell, and
  an O(1/h) inconsistency at radius h answers with an error `h²/r`: first
  order in the max norm on the innermost ring. Two of them are
  replaced on a ball:
  * `r ∂_r(v_tan/r)` in `e_rθ`, `e_rφ` is spelled `∂_r v_tan − v_tan/r`
    with the two-point derivative and the face average: the two-point
    difference of `g = v_tan/r` is exact on `v_tan ∈ {r, r², r³}` but
    33 % off across the first face on a field finite at the origin
    (`v_tan(0) = c` at `m = 0`, `|m| = 1`, where `g = c/r`), an O(1/h)
    shear that the Gram turns into O(1/h²) rows on the first rings; the
    new spelling is the unique two-point stencil exact on
    `v_tan ∈ {1, r}` (it equals the old one with the prefactor `r_k`
    replaced by `r_{c,k−1} r_{c,k} / r_k`), second order on the rest.
  * `Grad(Div)` (`build_staggered_grad_div`, order 2) differences a
    divergence whose radial flux form uses the face weights
    `r_{c,i−1} r_{c,i} = r_i² − h²/4` (`−h²/4` on the origin face, so the
    ring enters the first cell as `v_r(0)/h`) in place of `r_i²`
    (`_origin_flux_divergence`): the midpoint form is exact on `v_r = 1`
    and 33 % off on `v_r = r` in the first cell, and the gradient of that
    cell error is O(1/h) on the first face; the product form is exact on
    both in every cell. This block alone gives up exact self-adjointness
    near the origin; the saddle-point pair (G, Div) and the standalone
    Div keep the norm-adjoint form.
  The two-point difference of `v_tan/r` is exact to cubic order on the
  odd family, a 28-fold smaller error of the l = 1 toroidal decay rates
  (v_φ ∝ r near the origin) at 24×48 (§6) that the stencil exact on the
  origin constant does not have.
* **Row premultiply.** On a ball every compressible row is scaled by
  `r^{n}` to cancel the origin `1/r^n`; the origin ring now uses the
  half-cell radius `r_center[0]` instead of `r_edge[0] = 0`.
* **Midpoint volumes.** `W_p = r_c² Δr` is the midpoint rule for
  `∫ r² dr`; in the first cell it is 25 % low (3/4 of the exact `h³/3`),
  O(1/(12 i²)) in cell i. The first-cell row of the scalar Laplacian is
  O(1) inconsistent (∇²r² evaluates to 8, not 6) but the cell has volume
  O(h³), and every eigenvalue benchmark below converges at 2nd order.
  Exact cell volumes would move the certified shell operator and were not
  adopted.
* **Poles.** The θ-edge rows at θ = 0, π carry zero sinθ weight, so the
  pole `v_θ` DOFs are not degrees of freedom: for `|m| ≠ 1` the tangential
  components vanish there, and for `|m| = 1` they are finite and the
  strain Gram and the primal curl read them from the adjacent interior
  edge (`_pole_vtheta_prolongation`) rather than from the DOF; the pole
  closure of `curl_r` needs the finite pole value
  `(sinθ v_φ)' − i m v_θ = 0`, which the pinned DOF would break by
  `i m v_θ(0)`, an O(1) pole row that the dual curl differentiates into
  O(1/Δθ) first θ-centre rows. The assembler
  pins them explicitly, a Dirichlet row with no mass
  (`_pole_face_rows`, reported as `constraints['pinned']`); left alone the
  A row was empty against a unit mass and the pencil read `0 = λ x`,
  a `λ = 0` cluster of 2·Nr_c that a boundary-value problem did not
  constrain at all.

## 3. Structural properties verified (8×16 and 12×24, m = 0 … 3)

| property | result |
|---|---|
| `Cp ∘ G = 0` on the weightful curl grid | < 1e-11 (pole rows are weightless) |
| `W_v G + (W_p Div)ᴴ = 0` | 3e-17 (adjoint by construction) |
| Coriolis r–φ pair W-skew | 1e-18 |
| Coriolis θ–φ pair W-skew, seed recipe | defect 0.1–0.2 of the block (cosθ on the θ-edge vs θ-center); identical on the shell |
| Coriolis θ–φ pair, `coriolis_pairing='center'` (default; cosθ on the θ-centers in both blocks, partner the exact W-adjoint) | 1e-15; inviscid spectrum imaginary to 1e-14; r-mode error 1.6e-4 at 24×24 vs 3.4e-3 with the seed recipe (real parts up to 3e-3 on 8×16, 1.7e-3 on 12×24, 7.7e-3 on the 8×16 shell) and 6.9e-3 with the edge-cos 'adjoint' pairing |
| strain Gram (ρ₀ = e^{−r²}, stress-free), ρ₀W-self-adjoint, NSD | 2e-16, max eigenvalue 5e-16 (m = 0 rigid rotation) |
| scalar Laplacians, Neumann and Dirichlet, self-adjoint, NSD | 1e-16, max eigenvalue ≤ 0 (m = 0 Neumann constant) |
| compressible pencil rows empty in A and B | none (was 1 for m = 0, ±1) |
| m = 0 inertial pencil regular | residuals 1e-13 (was 0.36: constant-p null vector) |

## 4. Defects fixed

* the compressible `r^n` premultiply zeroed the origin anchor row of A
  and B at m = 0, |m| = 1 (review item 6); the origin ring now takes the
  half-cell radius (`staggered_general._ball_premultiply`).
* `build_staggered_scalar_laplacian` (ball branch): the outer-wall row of
  the gradient was the one-sided clamped Lagrange stencil (flux leak,
  non-self-adjoint, eigenvalue +0.28 at 8×16). Both scalar Laplacians
  now share `_radial_flux_stencils` with the origin-parity row (review
  item 19; moot in the flux form since the face has zero area) and a
  `wall_bc` ('neumann' / 'dirichlet', mirror ghost with a half-weight
  face).
* Scalar Laplacians at order > 2 were non-self-adjoint at every geometry
  (m = 0 growing mode +2e-2 on the shell); the divergences are now
  `−Gᵀ`. Interior rows keep 4th/6th-order consistency; the pole- and
  wall-adjacent rows are weakly consistent only and the eigenvalues
  converge at 2nd order at order 4 (the strong-consistent pair gave
  4th-order eigenvalues but complex pairs at m = 0, max |Im λ| = 60).
* The 1-vector + 1-scalar route silently dropped every term but Coriolis,
  Grad and Div, took |b| for the Coriolis sign and never checked the
  axis. `_parse_incompressible_pair` reads
  `a·Dt(v) + b·Cross(Ω, v) + g·Grad(p) + e·Lap(v) + d·Grad(Div(v)) = 0`,
  requires Ω = ẑ and a single Div(v) continuity row, raises on anything
  else, and supports the viscous pair with stress-free / no-slip walls.
* m = 0 inertial pencil: constant p was a null vector (singular pencil);
  the continuity row of cell 0 is replaced by the gauge p = 0 (the rows
  are dependent at m = 0).
* On a ball `_curl_radial_diff` ignored `wall_bc` at the outer wall.
* The curl-curl (Lamé) viscous force never imposed stress-free walls: the
  natural condition of the Gram `Cpᵀ W_c Cp` is n×(∇×v) = 0
  (`∂_r(r v_tan) = 0`; the measured toroidal rates were exactly the roots
  of that condition). The compressible path without the strain term and
  the incompressible viscous pair now use the unit-density strain Gram,
  whose natural condition with zeroed wall shear rows is zero traction.
* No-slip in `build_staggered_strain` applied no closure; it now uses the
  mirror ghost of `v_tan/r` with half-weight wall faces (2nd order).
* `|m| = 1` strain Gram: the pole `v_θ` DOFs, zero in every eigenmode,
  entered the first θ-center's strain (O(1/Δθ) spurious stress; toroidal
  rates at m = 1 did not converge on ball or shell).
* The entropy Dirichlet wall condition of the system was silently
  ignored by the thermal operator; `_entropy_wall_bc` wires it.
* The folded origin master had no viscous row (m = 0: an empty A row
  against a unit mass, the amplitude pinned to zero; |m| = 1: the
  `−(1/3) Grad Div` remainder alone, growing like 1/h), and the strain's
  `r ∂_r(v_tan/r)` was O(1/h) on the first face for a field finite at
  the origin, so every vector operator on a ball stalled at O(1) error
  at |m| = 1 and at first order at m = 0, |m| = 2 (the Grad Div face
  row) in `docs/OPERATOR_COVERAGE.md`; the ring weight, the strain
  spelling and the product face weights of §2 bring all 44 staggered
  ball cells to order 1.8–2.2.
* `Curl(Curl)` at |m| = 1 read the pinned pole `v_θ` in the pole closure
  of `curl_r` (shell and ball); the primal curl now reads it through the
  prolongation.
* `Div(a Grad(u) + b Transpose(Grad(u)))` with a ≠ b divided the
  nine-component gradient map, whose θφ and φθ components carry finite
  pole limits that only their sum (the strain) closes: O(1/Δθ) on the
  pole-adjacent ring at m = 0. It is built as `a Lap + b Grad Div` now.
* `Div(T)` of a non-symmetric tensor unknown (the nine components) was
  first order at m = 0 through the pole-adjacent ring and, through the
  adjoint of its one-sided wall rows, O(1/h) one cell inside the walls:
  the pole rows now carry the parity limits of the components, the
  divergence reads the pole share of the φθ component, and the wall
  rows carry the strain's no-slip closure; order 2.05, 1.92, 1.93 at
  m = 0, 1, 2 on the ball (2.07, 2.04, 2.01 on the shell).
* `Dot(B0, Grad(u))`, `Dot(B0, tensor(u))`, `Dot(B0, T)` refused a
  θ-dependent B0; they are the Hodge-paired contraction now, order
  1.95 to 2.07 on the ball, and `Curl(Cross(u, B0))` on a no-slip u is
  a plain face unknown (order 1.98 to 2.09).
* Certified shell operator: the compressible pencils with the strain
  term, inviscid, and MHD at m = 0, 1, 8, the inertial pencils at m = 1, 2
  and the order-2 Laplacians are bit-identical to the pre-edit snapshot.
  Moved on the shell, all uncertified: the inertial m = 0 pencil (gauge
  row), the Lamé-path pencil (strain Gram at unit density), the
  |m| = 1 strain pencil (pole fold), and the order > 2 Laplacians.

## 5. Found, not fixed

* The seed Coriolis θ–φ pairing is skew only to O(Δθ) on shell and ball
  (real parts of a few 1e-3 in the inviscid band). The exact partner is
  one line (`_coriolis_partner`, opt-in `coriolis_pairing='adjoint'`) but
  moves the certified shell operator, so the default stays 'seed'.
* Midpoint cell volumes (O(1) first-cell inconsistency, §2); the
  first-face row of the scalar Laplacian and of Div keep it (an O(h²)
  error, the cell having volume h³), only the blocks of §2 that
  differentiate a first-cell value again are re-weighted.
* At even |m| ≥ 2 the pole-adjacent `v_φ` ring of the strain Gram is
  O(1/Δθ) inconsistent (`e_θφ` has a finite pole limit and the
  sinθ-weighted average behind its cot term loses the pole edge), so
  `Lap(vector)` at |m| = 2 is first order between N = 24 and 48 on shell
  and ball alike (2.02 then 1.12). No pole weight or row repairs it
  inside the norm adjoint (docs/staggered_general_design.md §2b);
  `pole_closure=True` (`compile_fd2_staggered`) adds the pole share
  explicitly and restores order 2 (shell 2.02, 2.01; ball 1.98, 2.00
  over 12, 24, 48) at the price of the exact W-self-adjointness on that
  ring, and moves the certified m = 8 shell pencil, so it is off by
  default.
* The contraction `Dot(B0, Grad v)` leaves the origin ring rows empty
  (the φ-row components carry 1/r and a regular `B0_φ` vanishes at the
  origin), so a viscous fold at m = 0, |m| = 1 carries the Coriolis
  pair's ring entries but not the advection's; the inviscid anchor row
  is receive-only and decoupled, and the Doppler shift is exact.
* The m = 0 / |m| = 1 inviscid origin anchor is receive-only, so its row
  has no self-coefficient and contributes one decoupled `λ = 0`
  eigenvalue. (The 2·Nr_c pole `v_θ` DOFs no longer add a `λ = 0`
  cluster: they are pinned explicitly, §2.)
* Order > 2 scalar Laplacians converge at 2nd order in eigenvalues.
* CT MHD on the ball: the origin closure of §7. The velocity's origin
  ring receives no Lorentz force (`F_r` lives on the interior r-edges),
  so a viscous fold at m = 0, |m| = 1 carries the strain's ring rows and
  not the Lorentz force's, as it does not carry the advection's; the
  folded origin master of a vector potential is empty in A (the dual
  curl's origin rows are guarded, the origin face has no weight in the
  gradient's adjoint) and adds one decoupled `λ = 0` eigenvalue; and the
  first-centre averages of the Lorentz force close the origin by the
  parity of the equatorially symmetric family, O(h) on that one ring for
  the other family, as the strain does. A plain face unknown under
  `Curl(Cross(u, B0))` (a no-slip u advecting itself) compiles through the
  same CT blocks divided by the face areas and converges at order 2.

## 6. Benchmarks (`tests/test_ball_staggered.py`)

Zhang et al. (2001) inertial modes, seed pairing, ω = −Im λ, rel. error
against the curated targets; all 20 targets at m = 1, 2 are recovered on
16×32 within 8.4e-3, on 24×48 within 2.4e-3.

| m | RMS rel. error 16×32 | 24×48 | order |
|---|---|---|---|
| 1 | 2.2e-3 | 7.7e-4 | 2.6 |
| 2 | 3.1e-3 | 7.1e-4 | 3.7 |

The inviscid pencil enters none of the origin closures of §2 (no strain,
no Grad Div, the ring rows replaced by the regularity rows), and the
census is bit-identical before and after them.

Thermal decay ∂t s = Ek ∇²s, rate Ek k²:

| wall | m | l | k | rel. error 12×24 | 24×48 | order |
|---|---|---|---|---|---|---|
| Dirichlet | 0 | 0 | π | 1.75e-3 | 4.39e-4 | 2.00 |
| Dirichlet | 0 | 1 | 4.493409 | 1.74e-3 | 4.34e-4 | 2.00 |
| Dirichlet | 1 | 2 | 5.763459 | 3.92e-3 | 9.80e-4 | 2.00 |
| Dirichlet | 2 | 2 | 5.763459 | 2.24e-3 | 5.57e-4 | 2.01 |
| Neumann | 0 | 1 | 2.081576 | 1.10e-3 | 2.77e-4 | 2.00 |
| Neumann | 1 | 2 | 3.342094 | 1.93e-3 | 4.85e-4 | 1.99 |
| Neumann | 2 | 2 | 3.342094 | 8.87e-4 | 2.20e-4 | 2.01 |

Toroidal viscous decay, Ek = 1e-2, rate Ek k² with the stress-free
condition k j_{l−1}(k) = (l+2) j_l(k) and the no-slip condition j_l(k) = 0:

| wall | m | l | k² | rel. error 12×24 | 24×48 | order |
|---|---|---|---|---|---|---|
| stress-free | 0 | 1 | 33.2175 | 1.0e-2 | 2.6e-3 | 2.0 |
| stress-free | 0 | 2 | 6.2557 | 4.3e-3 | 1.1e-3 | 2.0 |
| stress-free | 1 | 1 | 33.2175 | 1.0e-2 | 2.6e-3 | 2.0 |
| stress-free | 1 | 2 | 6.2557 | 7.9e-4 | 1.9e-4 | 2.0 |
| stress-free | 2 | 2 | 6.2557 | 6.5e-3 | 1.6e-3 | 2.0 |
| stress-free | 2 | 3 | 14.9359 | 4.0e-3 | 1.0e-3 | 1.9 |
| no-slip | 0 | 1 | 20.1907 | 7.3e-3 | 1.8e-3 | 2.0 |
| no-slip | 0 | 2 | 33.2175 | 1.0e-2 | 2.5e-3 | 2.0 |
| no-slip | 1 | 1 | 20.1907 | 7.3e-3 | 1.8e-3 | 2.0 |
| no-slip | 1 | 2 | 33.2175 | 8.2e-3 | 2.0e-3 | 2.0 |
| no-slip | 2 | 2 | 33.2175 | 6.2e-3 | 1.5e-3 | 2.1 |
| no-slip | 2 | 3 | 48.8312 | 7.6e-3 | 1.8e-3 | 2.1 |

The l = 1 modes have `v_φ ∝ r` near the origin, on which the two-point
difference of `v_tan/r` is exact to cubic order (9.0e-5 at 24×48,
order 2.4); the stencil exact on the origin constant (§2) carries the
plain O(h²) error there, the price of the even family's closure.

Composite wall rows (`TestCompositeWallsBall`, docs/staggered_general_design.md §5),
all through `Problem.compile(discretization='staggered')` at order 2.
Robin, ∂t p = ∇²p with `dot(r_hat, grad(p)) + h*p = 0`, rate k² with
k j_l'(k) + h j_l(k) = 0 (the mirror-ghost flux closure with the Robin
coefficient):

| h | m | l | k | rel. error 12×24 | 24×48 | order |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | π/2 | 5.60e-4 | 1.40e-4 | 2.00 |
| 1 | 0 | 1 | 2.743707 | 2.12e-3 | 5.31e-4 | 2.00 |
| 1 | 1 | 1 | 2.743707 | 6.36e-4 | 1.59e-4 | 2.00 |
| 1 | 1 | 2 | 3.870239 | 3.42e-3 | 8.58e-4 | 2.00 |
| 1 | 2 | 2 | 3.870239 | 1.18e-3 | 2.94e-4 | 2.00 |
| 1 | 2 | 3 | 4.973420 | 4.04e-3 | 1.01e-3 | 2.00 |
| 3 | 0 | 0 | 2.288930 | 2.20e-4 | 5.67e-5 | 1.96 |
| 3 | 0 | 1 | 3.405608 | 3.22e-3 | 8.03e-4 | 2.00 |
| 3 | 1 | 1 | 3.405608 | 2.07e-3 | 5.16e-4 | 2.01 |
| 3 | 1 | 2 | 4.493409 | 5.20e-3 | 1.30e-3 | 2.00 |
| 3 | 2 | 2 | 4.493409 | 3.33e-3 | 8.31e-4 | 2.00 |
| 3 | 2 | 3 | 5.566646 | 6.34e-3 | 1.58e-3 | 2.00 |

Navier slip, ∂t v = Ek ∇²v with `dot(t_hat, v) + ls*dot(t_hat, dot(strain, r_hat)) = 0`
on the θ and φ rows, ls = 1/2, Ek = 1e-2: toroidal rate Ek k² with
(1 − ls) j_l(k) + ls k j_l'(k) = 0 (the strain closure with the slip length;
ls = 0 and ls = ∞ are the no-slip and stress-free pencils bitwise):

| m | l | k² | rel. error 12×24 | 24×48 | order |
|---|---|---|---|---|---|
| 0 | 1 | 7.5279 | 3.37e-3 | 8.45e-4 | 2.00 |
| 0 | 2 | 14.9787 | 6.23e-3 | 1.56e-3 | 1.99 |
| 1 | 1 | 7.5279 | 3.37e-3 | 8.45e-4 | 2.00 |
| 1 | 2 | 14.9787 | 3.79e-3 | 9.50e-4 | 2.00 |

Normal stress, the traction-free Stokes ball ∂t v = ∇²v − ∇p, ∇·v = 0 with
`dot(r_hat, dot(strain, r_hat)) - c*p = 0` and the two tangential stress rows
(the wall v_r faces stay degrees of freedom, the m = 0 gauge is lifted): the
poloidal roots of the 2×2 Bessel determinant of tests/test_dedalus_ball.py
and the toroidal family k j_l'(k) = j_l(k), rate k²:

| c | m | family | l | k² | rel. error 12×24 | 24×48 | order |
|---|---|---|---|---|---|---|---|
| 1 | 0 | pol | 2 | 7.1055 | 9.00e-3 | 2.29e-3 | 1.98 |
| 1 | 0 | pol | 2 | 29.9542 | 4.26e-3 | 1.09e-3 | 1.97 |
| 1 | 0 | pol | 3 | 16.0155 | 2.00e-2 | 5.02e-3 | 1.99 |
| 1 | 0 | tor | 2 | 6.2557 | 4.32e-3 | 1.09e-3 | 1.99 |
| 1 | 1 | pol | 2 | 7.1055 | 3.32e-3 | 8.97e-4 | 1.89 |
| 1 | 1 | pol | 2 | 29.9542 | 1.19e-3 | 3.28e-4 | 1.86 |
| 1 | 1 | pol | 3 | 16.0155 | 1.27e-2 | 3.25e-3 | 1.97 |
| 1 | 1 | tor | 2 | 6.2557 | 7.90e-4 | 1.94e-4 | 2.03 |
| 2 | 0 | pol | 2 | 8.0000 | 8.21e-3 | 2.07e-3 | 1.99 |
| 2 | 0 | pol | 2 | 28.2111 | 1.95e-3 | 5.11e-4 | 1.93 |
| 2 | 0 | pol | 3 | 18.7981 | 1.81e-2 | 4.57e-3 | 1.99 |
| 2 | 0 | tor | 2 | 6.2557 | 4.32e-3 | 1.09e-3 | 1.99 |
| 2 | 1 | pol | 2 | 8.0000 | 2.00e-3 | 5.50e-4 | 1.86 |
| 2 | 1 | pol | 2 | 28.2111 | 8.07e-4 | 1.70e-4 | 2.25 |
| 2 | 1 | pol | 3 | 18.7981 | 1.04e-2 | 2.68e-3 | 1.96 |
| 2 | 1 | tor | 2 | 6.2557 | 7.90e-4 | 1.94e-4 | 2.03 |

Free magnetic decay in a perfectly conducting ball (§7), rate k² with the
toroidal condition `(x j_l(x))' = 0` and the poloidal condition `j_l(k) = 0`
(`TestMagneticDecayBall`), the flux formulation (`b`, constrained transport)
and the vector potential (`A`, Coulomb gauge):

| m | family | l | n | k² | rel. error `b` 12×24 | 24×48 | order | `A` 12×24 | 24×48 | order |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | tor | 1 | 0 | 7.52793 | 5.80e-4 | 1.57e-4 | 1.88 | 1.89e-3 | 4.73e-4 | 2.00 |
| 0 | tor | 1 | 1 | 37.4148 | 1.33e-2 | 3.34e-3 | 2.00 | 1.26e-2 | 3.15e-3 | 1.99 |
| 0 | pol | 1 | 0 | 20.1907 | 8.54e-3 | 2.14e-3 | 2.00 | 1.83e-3 | 4.51e-4 | 2.02 |
| 0 | pol | 2 | 0 | 33.2175 | 1.17e-2 | 2.94e-3 | 2.00 | 2.40e-3 | 5.58e-4 | 2.10 |
| 1 | tor | 1 | 0 | 7.52793 | 2.62e-3 | 6.55e-4 | 2.00 | 1.60e-3 | 4.03e-4 | 1.99 |
| 1 | tor | 1 | 1 | 37.4148 | 1.50e-2 | 3.77e-3 | 1.99 | 1.23e-2 | 3.09e-3 | 1.99 |
| 1 | pol | 1 | 0 | 20.1907 | 7.51e-3 | 1.88e-3 | 2.00 | 6.90e-3 | 1.73e-3 | 1.99 |
| 1 | pol | 2 | 0 | 33.2175 | 9.14e-3 | 2.29e-3 | 2.00 | 1.24e-2 | 3.15e-3 | 1.98 |
| 2 | tor | 2 | 0 | 14.9787 | 2.21e-3 | 5.52e-4 | 2.00 | 1.07e-3 | 2.65e-4 | 2.01 |
| 2 | tor | 2 | 1 | 55.3995 | 1.51e-2 | 3.79e-3 | 2.00 | 1.21e-2 | 3.03e-3 | 2.00 |
| 2 | pol | 2 | 0 | 33.2175 | 7.47e-3 | 1.87e-3 | 2.00 | 4.03e-3 | 1.00e-3 | 2.01 |
| 2 | pol | 3 | 0 | 48.8312 | 1.01e-2 | 2.52e-3 | 2.00 | 6.71e-3 | 1.66e-3 | 2.02 |

The Malkus hydromagnetic inertial modes at m = 2 (`tests/test_analytic_evp.py`
benchmark 5, Le = 0.1 and 0.3, six targets each) land within 2.8e-3 of the
closed-form branches at 20×32 (against the frequency unit 2Ω = 2), RMS
order 2.28 and 2.24 over 10×16 / 20×32, with the product relation
ω₊ ω₋ = −Le² m² exact to 1e-6.

Compressible ball (regular polytrope-like background, §`make_regular_ball_system`):
with δ = 0, Ek = 0, β = 1e-3 the inertial modes agree with the
incompressible assembler to 1e-4 at m = 0, 1, 2 on 12×24 (matched modes
2.6e-6 to 5.4e-5 from their inertial partners against 2.1e-3 to 2.4e-3
for random control frequencies, unchanged by the origin closure, which
the inviscid path does not enter); the viscous convective ridge peak at
Ek = 3e-3, δ = 2e-5, m = 1 converges over 8×16 / 12×24 / 18×36
(Im ω = 0.06968, 0.06663, 0.06516; Richardson order 1.79).

## 7. Constrained transport on the ball: the origin closure

The involution unknown b is stored as face fluxes (`CTGrid`: `Φ_r` on
the r-edge ring `(r_edge, θ_c)`, `Φ_θ` on `(r_c, θ_edge)`, `Φ_φ` on the
centres) and its evolution as the incidence curl of edge circulations,
`dt Φ = −∮ E·dl` face by face. On a ball the first cells `(r ∈ [0, h],
θ ∈ [θ_j, θ_{j+1}])` have one degenerate face and two degenerate edge
families, and the closure of the origin is the geometry of those.

**Faces.** The inner r-face of the first cell sits at `r = 0` and has
area `r² sinθ Δθ = 0`, so the flux of a bounded field through it is
identically zero: `Φ_r(0, θ_j) ≡ 0` for every m. The two θ-faces
(area `∫₀ʰ r dr sinθ_k = (h²/2) sinθ_k`, the code's `r_c h sinθ_k`
exactly) and the φ-face (area `(h²/2) Δθ`, again exact) are ordinary.
Gauss on the first cell is therefore

    Div Φ (0, j) = Φ_r(h, j) + Φ_θ(r_c0, j+1) − Φ_θ(r_c0, j) + i m Φ_φ(r_c0, j),

with no radial inflow: the whole radial flux of the origin cell crosses
the first ring of faces at `r = h`. This is the divergence `CTGrid.divergence`
already builds (`−Φ_r(0) + Φ_r(h)` with `Φ_r(0) = 0`).

**Edges.** The edges of the first cell are the four r-edges from the
origin to `r = h` (length h, circulation `C_r = h E_r(r_c0, θ_k)`, the
midpoint rule on a genuine edge), the two θ-arcs at `r = h` (length
`h Δθ`) and the two φ-arcs at the corners `(h, θ_k)` (length `h sinθ_k`
per unit azimuth), and the degenerate ones: the θ-arc at `r = 0`
(length `r Δθ = 0`, `Len_th[0] = 0`) and the two φ-arcs at `r = 0`
(length `r sinθ = 0`, `Len_ph[0, :] = 0`), all of which collapse onto
the origin point. A circulation is a line integral, so on a degenerate
edge it vanishes whatever the value of E there: `C_θ(0, j) = 0`,
`C_φ(0, k) = 0`. The m-dependent limit of `E_θ` as `r → 0` (the
Cartesian vector `E(0)`: `E_θ(0, θ) = −E_z sinθ` at m = 0, `E_⊥ cosθ`
at |m| = 1, zero beyond) is never needed: it multiplies a zero length.
Faraday on the three faces of the first cell that touch the origin is
then

    dt Φ_r(0, j)     = −[C_φ(0, j+1) − C_φ(0, j)] + i m C_θ(0, j) = 0,
    dt Φ_θ(r_c0, k)  = −i m C_r(r_c0, k) + C_φ(h, k) − C_φ(0, k) = −i m C_r(r_c0, k) + C_φ(h, k),
    dt Φ_φ(r_c0, j)  = −[C_θ(h, j) − C_θ(0, j)] + [C_r(r_c0, j+1) − C_r(r_c0, j)]
                     = −C_θ(h, j) + C_r(r_c0, j+1) − C_r(r_c0, j),

each the exact Stokes circulation of its face with the degenerate edge
contributing nothing, and the origin ring's tendency vanishes
identically, consistent with its value.

**The origin ring is not a degree of freedom.** `Φ_r(0, θ_j)` is zero
and stays zero, so the ring is pinned for every m (a Dirichlet row with
no mass in the inviscid path, dropped by the symmetric reduction
otherwise), the way the wall rings of a flux unknown are. There is no
fold: the velocity's origin ring is a point value `v_r(0, θ_j)`, Nθ
copies of one Cartesian vector, and its regularity rows tie the copies
together; a flux through a zero-area face carries no value to tie.

**Div Curl = 0 survives exactly.** The identity is combinatorial: the
sum over the six faces of a cell of `∮ E·dl` traverses every edge of the
cell twice with opposite orientation, the degenerate edges included
(they contribute zero twice). On the first cell,

    Div (dt Φ)(0, j) = dt Φ_r(h, j) + dt Φ_θ(j+1) − dt Φ_θ(j) + i m dt Φ_φ
      = [−C_φ(h, j+1) + C_φ(h, j) + i m C_θ(h, j)]
        + [−i m C_r(j+1) + C_φ(h, j+1)] − [−i m C_r(j) + C_φ(h, j)]
        + i m [−C_θ(h, j) + C_r(j+1) − C_r(j)] = 0

term by term, with no closure value entering, so `_check_div_curl`
passes on the ball to rounding of the block maximum at every m (it is
asserted at m = 0, 1, 2, 8 in `tests/test_staggered_general.py`).

**Regularity of the first ring of fluxes.** A Cartesian-regular field
has `b(0) = c`, `b_r(r, θ) = c·r̂(θ) + O(r)`, so the first ring is
`Φ_r(h, θ_j) = h² ∫ b_r(h, θ) sinθ dθ ~ c_z cosθ` at m = 0, `~ c_⊥ sinθ`
at |m| = 1, `O(h³)` at |m| ≥ 2 -- to leading order only. The `O(h³)`
remainder carries every angular order the field's Taylor expansion
admits (`b_r ~ r P_2(cosθ)` of an l = 2 poloidal field at m = 0, for
instance), so regularity imposes no discrete constraint on the first
ring beyond the divergence of the first cells, which the involution
preserves exactly. Its one closed consequence is at m = 0, where summing
the first-cell divergences over j telescopes the θ-fluxes (the pole
faces have zero area) to `Σ_j Φ_r(h, j) = 0`: the net flux out of the
sphere `r = h` vanishes, as the `cosθ` pattern does; at m ≠ 0 the
φ-flux `i m Φ_φ` enters and there is no sum rule. So the flux
formulation carries no regularity rows on a ball at any m: the first
cells' divergence is the regularity statement, and it is an identity of
the evolution.

**The physical field near the origin.** The blocks that read
`b = Φ / area` or `j = curl b` are the resistive term and the Lorentz
force, and only the second needs a closure.

* *Resistive term* `dt Φ = −curl(η j)`, j on the edges. The circulations
  on the degenerate edges vanish by length, so the origin values of j
  are never used; the origin-adjacent edges of nonzero length take
  `j_r(r_c0, θ_k)` from `b_φ(r_c0, ·)` and `b_θ(r_c0, ·)` (no radial
  stencil), `j_θ(h, j) = i m b_r(h, j) / (h sinθ_j) − [r_c1 b_φ(r_c1) −
  r_c0 b_φ(r_c0)] / h²` and `j_φ(h, k) = [r_c1 b_θ(r_c1) − r_c0
  b_θ(r_c0)] / h² − ∂_θ b_r(h, ·) / h`, the two-point difference of
  `r b_tan` across the face `r = h` from the centres `h/2` and `3h/2`:
  centred, second order, and its window never reaches `r < 0`, so the
  origin parity has nothing to close. The ball's resistive blocks are the
  shell's.
* *Lorentz force* `F = pre (j × B0)` on the velocity faces, j brought
  from the edges by the plain two-point averages of the constrained-
  transport recipe. `F_r` on the interior r-edges reads j at `r ≥ h`
  only. `F_θ` on the first θ-edges reads `j_φ(r_c0, k) = ½ [j_φ(0, k) +
  j_φ(h, k)]` (`K_to_vth`), `F_φ` at the first centres reads
  `j_θ(r_c0, j) = ½ [j_θ(0, j) + j_θ(h, j)]` (`rf_to_c`), and the
  background-current block `J0 × b` reads `b_r(r_c0, j) = ½ [b_r(0, j) +
  b_r(h, j)]`: three radial averages onto `r_c0 = h/2` whose origin node
  is a zero-length edge, a zero-area face, or a corner where `1 / r` is
  guarded, so the plain average reads it as zero. Their regular limits
  are the Cartesian vector's components: `j_θ(0, θ) = −j_z sinθ` (m = 0)
  or `j_⊥ cosθ` (|m| = 1), `j_φ(0, θ) = j_⊥'` (|m| = 1), `b_r(0, θ) =
  b_z cosθ` (m = 0) or `b_⊥ sinθ` (|m| = 1), all zero for |m| ≥ 2; zero
  is an O(1) error on the first ring at m = 0 and |m| = 1 (the ring has
  volume O(h³), so the eigenvalues stay second order, but the innermost
  ring of the eigenfunction is wrong). The closure is the velocity's:
  the origin node is the reflection of the first interior node with the
  vector parity `p = (−1)^{max(|m|−1, 0)}` of §1 (`_origin_parity(m,
  "vector")`), and the two-point Lagrange interpolation to `h/2` on the
  extended nodes `{−h, h}` gives

      j(r_c0) = ¼ (3 + p) j(h):   j(h) for p = +1 (m = 0, |m| = 1),   ½ j(h) for p = −1,

  which is the |m| = 1 pole prolongation's form -- a field finite at a
  degenerate node reads the adjacent interior node, `j(0) := j(h)`, then
  the plain average -- and for the odd parity is the guard's zero average
  (exact to O(h³) on a field ∝ r). On the equatorially symmetric family
  it is exact to the order of the interior stencil; on the other family
  it is O(h) on that one ring, the price the velocity's strain also pays.
  `CTGrid` closes the first row of its centre averages this way on a ball
  (`_origin_average`); on a shell the averages are untouched.
* *Induction blocks* need no origin value of v: every velocity
  interpolation onto a degenerate edge is multiplied by its zero length.

**What the velocity contributes.** The Lorentz force on the velocity's
origin ring is zero (`F_r` lives on the interior r-edges: no flux
crosses the origin face), so the ring stays receive-only in the
inviscid pencils and, in a viscous fold, the master row carries the
strain's ring rows and not the Lorentz force's, as it does not carry the
advection's (§5). The Malkus benchmark below is inviscid at m = 2, where
the ring is Dirichlet.

**Walls.** A conducting or radial wall on the flux unknown is the outer
wall only; `CTGrid.wall_masks` also zeroes the r = 0 circulations, which
are zero by length, so the masks are inert there. An involution unknown
with no wall condition at all (the Malkus problem, whose b = −(m/ω) v
inherits v's impenetrable wall through the induction equation) keeps
its wall fluxes as degrees of freedom, evolving with the unmasked wall
circulations of the ideal electric field (`_classify_walls`, kind
'free'); with a resistive term the wall current would need a closure the
one-sided corner stencils do not provide, and the free wall is refused
there.
