# The sh_fd core in the full sphere

The sh_fd core expands every unknown in spin-weighted spherical harmonics
at fixed m and keeps each (l, spin) coefficient as a function of r on a
Fornberg finite-difference grid.  In a shell the radial operators need
nothing beyond the stencils; in the ball (r_inner = 0) the coefficients
have prescribed behaviour at the origin and the 1/r, 1/r^2 factors of the
angular terms are singular there.  This note derives the regularity rules,
chooses the closure and records the benchmark conditions.

## 1. Regularity of the spin components at the origin

Let x = r x_hat with x_hat = (theta, phi).  A field is regular at the
origin when its Cartesian components are analytic (locally polynomial) in
x.  Continue every coefficient function to negative r by evaluating the
Cartesian field at -x while keeping the local basis (r_hat, theta_hat,
phi_hat) attached to the *direction* x_hat.  Two facts fix everything:

* the antipode: e_r(-x_hat) = -e_r(x_hat), e_theta(-x_hat) = e_theta(x_hat),
  e_phi(-x_hat) = -e_phi(x_hat); hence the spin basis
  e_pm = (e_theta pm i e_phi)/sqrt2 satisfies e_pm(-x_hat) = e_mp(x_hat);
* the harmonics: sY_lm(pi - theta, phi + pi) = (-1)^l (-s)Y_lm(theta, phi)
  in the phase convention of `angular.sYlm_theta` (checked numerically to
  6e-15 for m = 0..3, |s| <= 2, l <= 7).

**Scalar.**  f(-r, x_hat) = f(r, -x_hat) = sum_l f_l(r) Y_l(-x_hat), so

    f_l(-r) = (-1)^l f_l(r):        f_l(r) = r^l x (even series in r).

**Vector.**  u_r(-r, x_hat) = u(-x) . e_r(x_hat) = -u(-x) . e_r(-x_hat)
= -u_r(r, -x_hat), and u_+(-r, x_hat) = u(-x) . e_+(x_hat)
= u(-x) . e_-(-x_hat) = u_-(r, -x_hat).  With the harmonic identity

    u_{r,l}(-r) = (-1)^{l+1} u_{r,l}(r)      u_{r,l}(r) = r^{l-1} x even   (l >= 1),  r x even (l = 0)
    u_{pm,l}(-r) = (-1)^l u_{mp,l}(r)        u_{pm,l}(r) = r^{l-1} x (any series)

The spin components of a vector have *no* parity of their own: the
spheroidal part (u_- - u_+)/sqrt2 ~ r^{l-1} x even (gradient of a regular
scalar, e.g. u = e_z: u_theta = -sin theta at l = 1) and the toroidal part
(u_- + u_+)/sqrt2 ~ r^l x even (e.g. u = e_z x x: u_phi = r sin theta at
l = 1) carry opposite parities.  Vasil et al. (2019) resolve this with the
"regularity" basis, whose three components go as r^{l-1}, r^l, r^{l+1}; in
the spin basis the same content is the swap rule above.  Only the leading
power r^{l-1} is common to all three spin components (l >= 1).

**Rank-2 tensor** (the gradient of a vector).  Each r index contributes a
sign and each pm index a swap:

    T^{ab}_l(-r) = (-1)^{l + n_r} T^{a_bar b_bar}_l(r),   n_r = number of r indices,

so every component of grad(u) is r^{l-2} x (series) for l >= 2 and the
parity pattern is (-1)^{l + n_r} with the spin indices mirrored.  The
general rule for a rank-n tensor is (-1)^{l+n_r} with all pm mirrored, which
reduces to (-1)^{l+s} only for scalars and for the pm components of a
vector; the radial component of a vector is (-1)^{l+1}, not (-1)^l.

The rank-2 table, per component T^{ab} of total spin s = s_a + s_b (a
tensor unknown is carried on these nine blocks, section 5):

    component        spin s   parity under r -> -r      leading power
    T^{rr}             0      (-1)^l   onto itself      r^{l-2} (l >= 2), r^l (l < 2)
    T^{+-}, T^{-+}     0      (-1)^l   onto each other  r^{l-2} (l >= 2), r^l (l < 2)
    T^{r pm}          pm1     (-1)^{l+1} onto T^{r mp}  r^{l-2} (l >= 2), r^0 (l = 1)
    T^{pm r}          pm1     (-1)^{l+1} onto T^{mp r}  r^{l-2} (l >= 2), r^0 (l = 1)
    T^{pm pm}         pm2     (-1)^l   onto T^{mp mp}   r^{l-2} = r^{l-|s|}

A component exists only for l >= |s|: the spin pm1 blocks lack l = 0 and
the spin pm2 blocks lack l = 0, 1 (the harmonics vanish there and every
operator gives them zero coupling), and the compiler pins them (section 3).
For l >= 2 the leading power r^{l-2} is common to all nine components (the
Hessian of a harmonic polynomial of degree l has spin-0 content at l with
r^{l-2}); below l = 2 the parity rule and analyticity leave r^{l-|s|}.  Only
the parity enters the closure: the radial derivative of a component of
parity sigma lands on its mirror image with the folded stencil,
`_dr_blocks(..., rank=2)`, for `Div(T)` of a gradient or of a tensor
unknown alike.  The identities Div(Grad u) = Lap u and
Div(Grad u^T) = Grad(Div u) hold to roundoff on polynomial fields in the
ball at fd_order 6
(`tests/test_sh_fd_ball.py::test_operators_exact_on_polynomial_fields`),
which checks the tensor closure together with the connection terms of
section 5, and `test_rank2_regularity_at_the_origin` checks the spin pm2
components of Grad(grad(r^l (1 - r^2) Y_lm)), k^mu_{l,mu} k^mu_{l,0}
r^{l-2} (1 - r^2), against their closed derivative on a tensor unknown.

## 2. Closure: mirrored ghost nodes on a cell-centred grid

Option (ii) (a node at r = 0 with regularity rows) needs L'Hospital limits
of 1/r and 1/r^2 terms that differ per (l, spin) and, for the vector spin
components, per spheroidal/toroidal content that a single row cannot
separate.  Option (i) needs only the continuation rule and keeps every
operator a plain stencil, so it is the one implemented:

* nodes r_i = (i + 1/2) h, h = r_outer/(N_r - 1/2), i = 0..N_r-1: the last
  node is the wall (the BC rows replace it as in the shell), the first is
  h/2 > 0, and the mirrored set {-r_j} makes the extended grid uniform and
  symmetric so that every interior stencil is centred;
* for node i the Fornberg stencil of width fd_order + 1 is taken on the
  extended grid; weights on real nodes go to A1, A2 and weights on the
  ghost -r_j fold onto node j in G1, G2 (`radial_diff_matrices_ball`);
* the l-dependent derivative of a coefficient block is then
  D = kron(I_l, A) + kron(P, G) with P = diag(sigma_l), where the folded
  part acts on the mirrored component: sigma_l = (-1)^l for a scalar and
  for u_pm (landing on u_mp), sigma_l = (-1)^{l+1} for u_r
  (`ShFdCompiledProblem._dr_blocks`);
* 1/r and 1/r^2 are evaluated at the nodes only, all of which are > 0.

Since the continued coefficient functions are analytic through r = 0, the
extended-grid stencil has the ordinary truncation error O(h^p) (p =
fd_order) at every node; the 2/r D1 term multiplies that by 2/r_0 = 4/h at
the first node only, an O(1)-node contribution to the eigenvalue error of
O(h^p) after quadrature weighting.  The measured orders in
`tests/test_sh_fd_ball.py` confirm p.

Custom `nodes` are accepted in the ball when they are positive and end at
r_outer; the ghost set is always the mirror image.

Shell pencils are untouched: the ghost matrices are only built when
r_inner = 0, and the main-term expressions are the ones used before.

## 3. m = 0

`AngularVasil` accepts m = 0 with l = 0..ell_max.  Spin pm1 harmonics
do not exist at l = 0; the compiler pins those coefficients (K row = the
unit, M row = 0, so they become infinite eigenvalues) after the boundary
rows are placed.  Every operator already gives them zero coupling
(kcoef(mu, 0, pm1) = 0 and the quadrature matrices have zero rows and
columns there).  The same pinning applies to every component with
|spin| > l: the spin pm2 blocks of a tensor unknown at l = 0, 1 (m = 0) and
l = 1 (m = 1), `_absent(name, component)` = max(0, |s| - m) leading l's.

A scalar boundary row that lands on a spin pm1 component (the stress-free
tractions, `Dot(theta_hat, Dot(strain, r_hat))`) reaches the spin pm1
traction through multiplication by the constant theta_hat components,
whose matrix Q (spin pm1 -> 0) at m = 0 is a parity checkerboard with a
zero l' = 0 column: L + 1 output rows constrain L unknowns with rank L.
The compiler therefore places the rows l < ell_max on the slots l >= 1
(dropping the least-resolved row), which leaves both parity blocks square;
dropping the l = 0 row instead is singular for odd ell_max.  This applies
to the shell at m = 0 as well, and the shift is the number of absent l's
of the host block in general.

## 3a. Convergence

The eigenvalue error is C_p h^p + C_(p+1) h^(p+1) + ..., the second term
coming from the off-centred stencils of the last two nodes (order p - 1
locally, as in the shell).  The two terms have opposite signs for some
modes, so the error crosses zero at a mode-dependent resolution (N_r of
order 20-60 for the benchmarks) and a two-point order estimate is only
meaningful on the far side of that crossing; the tests measure the order
on the modes without a crossing in their pair and check accuracy on all.
Reference roots must carry full precision: a 7-digit k limits lambda = k^2
to 4e-7.

## 4. Benchmark conditions (unit ball)

* Scalar Helmholtz Lap(p) = lambda p, lambda = -k^2 with p = j_l(kr) Y_lm:
  Dirichlet j_l(k) = 0 (l = 0: pi, 2 pi; l = 1: 4.493409, 7.725252;
  l = 2: 5.763459); Neumann j_l'(k) = 0 (l = 1: 2.081576; l = 2: 3.342094;
  l = 0: 4.493409).
* Vector diffusion dt(v) = Ek Lap(v), stress-free wall.  A toroidal mode
  v = T(r) r_hat x grad_ang Y_lm has u_r = 0 and the Laplacian acts on T
  as the scalar radial operator, so T = j_l(kr) and lambda = -Ek k^2.  The
  tangential stress e_{r phi} = (r/2) d(u_phi/r)/dr vanishes when
  d(T/r)/dr = 0 at r = 1, i.e.

      k j_l'(k) = j_l(k)     (l = 1: 5.763459, 9.095011; l = 2: 2.501133, 7.136009; l = 3: 3.864700).

  The l = 1 roots coincide with the Dirichlet zeros of j_2 through the
  recurrence j_1' - j_1/r = -j_2.  The modes are divergence-free.
* Inviscid inertial modes dt(v) + 2 cross(ez, v) + grad(p) = 0, div(v) = 0,
  r_hat . v = 0 at r = 1: Zhang et al. (2001) frequencies at m = 1, 2 and
  the sectoral r-mode omega = -2/(m + 1).  `EvpResult.omega` is
  omega = -i lambda for x ~ exp(-lambda t) = exp(-i omega t); Zhang's
  exp(i(m phi + omega t)) makes omega_Zhang = -omega_EvpResult, and the
  r-mode is -2/(m + 1) in the EvpResult convention as in the shell test.
  All twenty Zhang targets are recovered to 4e-5 at N_r = 24,
  ell_max = m + 12, fd_order 4.
* Cross-core: the Dedalus core on the same Helmholtz ball problem
  (`DedalusGrid(N_r, N_theta, r_inner=0.0)`), eigenvalue by eigenvalue;
  the test skips while that core cannot compile a ball.

## 5. Tensors: Div(T) and tensor backgrounds

A rank-2 tensor is carried as the nine spin components
T^{ab} = e_a . T . e_b, a, b in (-, +, r), e_pm = (e_theta pm i e_phi)/sqrt2,
each of total spin s_a + s_b (block 3*ia + ib of the layout).  The
compiler builds them from a vector through `_grad_vector`, and a tensor
unknown occupies the same nine blocks (section 5a).

**Covariant derivative.**  Along e_mu (mu = pm) a component of total spin
s is differentiated by (1/r) k^mu_{l,s}, spin s -> s + mu, and every index
adds the connection of a vector index: the r component feeds the -mu
component with +1/r and the mu component feeds the r component with
-1/r (this is `_grad_vector`'s T^{mu,-mu} += u_r/r and
T^{mu,r} -= u_mu/r).  Along e_r there is no connection.

**Divergence.**  Contracting the derivative direction with the first index
(the pairing e_r e_r + e_+ e_- + e_- e_+ of the identity),

    (Div T)^b = d_r T^{rb} + 2 T^{rb}/r
              + (1/r) (k^-_{l,1+s_b} T^{+b} + k^+_{l,-1+s_b} T^{-b})
              - (T^{+-} + T^{-+})/r     (b = r)
              + T^{mu r}/r              (b = mu),

`_div_tensor`; the k coefficients with |s| = 2 vanish for l < 2 as the
spin-2 harmonics do.  The viscous stress Div(Grad v + Grad v^T) and the
one-sided Div(Grad v^T) therefore compile, and both converge at order 4 on
the shell and the ball in `docs/OPERATOR_COVERAGE.md`.

**Tensor backgrounds.**  `Background.Tensor` components T_{ij} in
(r, theta, phi) become the spin components above on the quadrature grid
(`_bg_field`); `Dot(T0, v)` is (T0 . v)^a = sum_b T0^{ab} v_{-b} and
`Dot(v, T0)` is (v . T0)^b = sum_a v_{-a} T0^{ab}, each entry a quadrature
multiplication between spins -s_b (or -s_a) and s_a (or s_b) that needs
the |s| <= 2 harmonics already built for the strain rows; `T0 * f` for a
scalar unknown f gives a tensor.  Not built: Dot of a tensor background
with a tensor expression (a rank-3 result) and Grad and Lap of a tensor
(rank 3 and the rank-2 Laplacian's connection).

**Tensor unknowns (5a).**  A rank-2 unknown T is nine blocks of the
layout, identity and Dt on them the unit, `Div(T)` the `_div_tensor`
above on the unknown's blocks (with the rank-2 ghost closure of section 1),
`Dot(B0, T)` and `Dot(T, B0)` the contractions of `_dot_B_X`, and Grad(u)
with its transposes the tensor-of-vector map into the same blocks.  The
constraint classifier (`spdee.ir.constraints.classify`, non-mimetic; the
record is consulted whenever the system is in its vocabulary and is
reported in `classification_error` otherwise) decides the placement: a
definition `T - tensor(u) = 0` never under Dt is eliminated at compile time
(`_lin(T)` returns -rest/c, T gets no blocks, the definition equation no
rows), so the pencil of `dt(v) - Ek div(T) = 0` is entrywise that of
`dt(v) - Ek div(grad(v) + trans(grad(v))) = 0` with the same spectrum to
rounding; `dt(T) + T - strain = 0` keeps T in the state with I on T and
-strain on v in K and I on T in M, the absent spin pm2 rows pinned
(`tests/test_sh_fd_parity.py`).  Boundary rows on a tensor unknown are not
hosted.

**Benchmark.**  dt(v) = Ek Div(Grad v + Grad v^T) with stress-free walls
is dissipative for compressible motion (the wall term of the energy
balance vanishes with v_r = 0 and zero tangential traction, unlike the
bare vector Laplacian): the spectrum is real and non-negative with the
l = 1 rigid rotations at zero, the toroidal rates are Ek k^2 with
k j_l'(k) = j_l(k) on the ball (and the shell analogue with y_l), and the
spheroidal rates agree with the collocated core
(`tests/test_sh_fd_backend.py`, `tests/test_sh_fd_ball.py`).

## 6. Wall rows, multipliers and the divergence cleaning

**Hosting.**  Every boundary row is translated directly (`_lin(bc.expr)`,
unit vectors and coefficients multiplied through the quadrature, so a
theta-dependent coefficient couples the l's like any background) and
replaces the wall-node rows over l of one block of one unknown.  A
single-shape row (Dirichlet, projection, Neumann, tensor projection,
curl-cross) goes on the block it acts on most strongly, as before; the two
rows of a stress-free pair have equal strength on the spin pm blocks to
rounding, so which spin block hosts the theta_hat row is a rounding-level
tie (a permutation of two rows, spectrum-preserving) that the certified
pencils fix.  A composite row (`BC_LINEAR_COMBINATION`: Robin, Biot, Navier
slip, the normal stress) goes on the record's host: the highest-rank
unknown, block +1 for a theta_hat row, -1 for a phi_hat row, r otherwise,
so the normal stress `dot(r_hat, dot(strain, r_hat)) - c p = 0` is one row
on v's radial wall slot with the columns of both v and p, whatever the
sizes of the two columns.  A theta / phi pair imposes the two spin
components of one tangential condition exactly (the pair of Galerkin rows
is a nonsingular map of the pair of spin rows); a lone tangential row or a
mixed one is accepted as the Galerkin projection of the wall function on
the spin-0 harmonics, which the spectral contract lists as refused.

Closed forms at fd_order 4 (`tests/test_sh_fd_parity.py`, N_r = 24 and 48,
errors at 48 and the RMS order): Robin k j_l'(k) + h j_l(k) = 0 on the
ball and the j_l / y_l determinant with k r f'(k r) + h r f(k r) at both
radii on the shell, m = 0, 1, 2: 2e-5, order 3.6-3.8; Biot on S with
kappa: the same roots scaled; Navier slip (1 - ls) j_l + ls k j_l' = 0
and the shell determinant with (r - ls) f + ls k r f' (ls = 1/2), m = 0,
1, 2: 1e-5, order 3.7, with ls = 0 the no-slip j_l(k) = 0 and ls = 1e8 the
stress-free k j_l' = j_l to 1e-5; the traction-free Stokes ball at m = 0,
1 and c = 1, 2: the toroidal family and the poloidal determinant of
`tests/test_dedalus_ball.py` to 1e-4 at order 3.5-4 (the l = 0 mode
lambda = 10/c of the tau method is the uniform expansion v = alpha r that
`div(v) + tau = 0` admits; a row-replacement core imposes div(v) = 0 at
every node but one and has no such mode).

**Multipliers.**  A scalar that appears only under Grad in a vector
equation (psi of the vector potential, `__div_b` below) has no scalar
equation; its rows are those of its constraint equation (the classifier's
`multipliers`, else the first Dt-free scalar equation not referencing it),
and its wall rows replace the constraint rows at the wall nodes.  Handing
them to the vector's radial block instead, the tau owner's slot of the
Dedalus core, drops the radial evolution row at the wall node and keeps
div(A) = 0 there: on the free decay of the vector potential in the
conducting shell the poloidal rates are then 77% off at N_r = 24 and 11%
at 48 (order 2.7) with psi of order 0.4 |A|, against 3e-3 and 2e-4 (order
3.7) with psi and div(A) at the discretisation error on the constraint
rows.  The vector potential `dt(A) - cross(v, B0) - eta lap(A) - grad(psi)
= 0`, `div(A) = 0` with tangential A = 0 and psi = 0 therefore compiles,
gives the conducting-wall rates on the shell and the ball at the stencil
order, and the 2 kG dipole Rossby shift of the solar eigenproblem in the A
formulation is -0.06889-0.00919j at N_r = 16, ell_max = 16 against
-0.07060-0.01039j in the b formulation at the same resolution
(-0.06957-0.01037j at 24 x 20, -0.06912-0.01035j at 32 x 24; the Dedalus
core -0.0686-0.0102j).

**Gauge rows.**  A declared gauge unknown gets the mean-zero row
sqrt(4 pi) int p_00 r^2 dr = 0 at m = 0 on the row of its constraint
equation at the first node inside the outer wall (cell weights in the
ball, the trapezoid rule in the shell); a gauge unknown with a wall row of
its own (psi = 0) has no constant left to fix and gets none, since the row
would replace a constraint row and admit a spurious l = 0 mode.

**Divergence cleaning.**  A written `div(b) = 0` with a curl evolution and
no multiplier is the classifier's involution; on this non-mimetic core the
record's `DivergenceCleaning` adds the scalar `__div_b` to the layout,
`+ grad(__div_b)` to the evolution equation, hosts the written row on its
blocks and pins it to zero at every wall the record names, on the
divergence rows at the wall nodes (b's radial slot holds b_r = 0).  Free
decay in the conducting shell and ball at m = 0, 1, 2 then gives the
toroidal d(rT)/dr = 0 and poloidal S = 0 families at 2e-5 (N_r = 48, order
3.7-4), with div(b) and the multiplier of the target modes at 1e-4 to
1e-5 and decreasing at the same order.  The record's mean-zero gauge at
m = 0 is not imposed on top of the Dirichlet walls: the walls already fix
the constant, and the extra row replaces one more divergence row and
admits a spurious l = 0 mode with b_r (measured at lambda = 8805 on the
shell and 2864 on the ball at N_r = 24).  An implied involution (an
induction equation without a written constraint: the Malkus and solar-eigenproblem MHD
systems) is left as it is; the record prescribes its cleaning too, but the
certified pencils carry none.
