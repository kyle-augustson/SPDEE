# Closed-form eigenvalue benchmarks

Reference for `tests/test_analytic_evp.py`.  Every benchmark is an
eigenvalue problem with an analytic spectrum, written once as a
`spdee.Problem` from strings and handed to all four numerical cores -- the
collocated fd2 core, the staggered (Arakawa-C) core, the sh_fd core and the
Dedalus core -- at two resolutions.  The point of the set is that it is not
tuned to the solar eigenproblem: it exercises Grad/Div, the Coriolis pair,
advection by a mean flow, buoyancy, induction and the Lorentz force against
values that come from a derivation rather than from another code.

This page carries the derivations, the parameter choices, the measured
numbers and the reason each tolerance is what it is.  Run

    OMP_NUM_THREADS=3 PYTHONPATH=. python -m pytest tests/test_analytic_evp.py -s

to reproduce the `BENCH` rows quoted below.

## Conventions

Every core assembles `K x = lambda M x` with the `Dt` terms in `M` and the
rest in `K`, so `omega = -i lambda` (`EvpResult.eigenvalues_to_omega`): a
decay rate `gamma > 0` appears at `lambda = +gamma` and an oscillation at
`lambda = i omega`.  Fields go as `e^{i m phi - i omega t}`.

Zhang et al. (2001) write their modes as `e^{i(m phi + 2 sigma t)}` and
tabulate `omega_Z = 2 sigma`, so their frequencies appear here at
`omega = -omega_Z`.  The sectoral Rossby mode fixes the same sign
independently: it sits at `omega = -2/(m+1)` on every core, which is what
`tests/test_dedalus_ball.py::test_rossby_family` and
`tests/test_sh_fd_ball.py::test_rmode` already assert.

## Method

Each (core, resolution) pair is solved once as a dense census through the
shift-invert resolvent, `spdee.solvers.dense(K, M, sigma=...)`, and every
analytic target of that benchmark is read off the one census.  ARPACK was
tried first and rejected: on these pencils, whose `M` has a large kernel
(boundary rows, gauge rows, constraint rows), `scipy.sparse.linalg.eigs`
stagnates or raises `ArpackNoConvergence` at some resolutions and shifts and
converges in under a second at others, with no setting that works
everywhere -- a 3456-unknown collocated acoustic pencil failed at
`(k=8, offset=0.05)` after 44 s and converged in 21 s at `(k=20, ...)` on
one grid and the other way round on the next.  The dense census is
deterministic, gives the whole spectrum (so mode counts and spacings are
available for free) and costs 1-40 s at the resolutions used here.

Saddle-point pencils (inviscid inertial, Malkus, gravity) are read through
the residual gate `resid < 1e-8`, as `tests/test_ball_staggered.py` does.
The collocated Grad/Div pair is not discretely adjoint, so its inviscid
censuses also carry non-neutral spurious pressure modes; those are dropped
by the `NEUTRAL = 1e-6` gate on `|Im omega|` (physical modes of an inviscid
problem are neutral to rounding).

## Resolutions

| core | coarse | fine | ratio | stencil order |
|---|---|---|---|---|
| fd2 (collocated) | 12 x 18 | 18 x 27 | 1.5 | 4 in r and theta |
| staggered | 10 x 16 | 20 x 32 | 2 | 2 |
| sh_fd | N_r 24, ell_max m+6 | N_r 48 | 2 | 4 |
| dedalus | 10 x 12 | 14 x 16 | -- | spectral |

The free-decay benchmark gives Dedalus `14 -> 20` radial modes instead: its
gravest rates sit at `k ~ 7`, which a shell of thickness 0.5 does not
resolve with 14 Chebyshev modes.  The collocated m = 2 inertial census runs
on `18 x 27 -> 27 x 40` (ratio 1.5), where the modes are identified by
eigenvector (section 3).

## Reading the order

A two-grid order is only meaningful while the discretisation error is
dominated by one term.  Below a core-dependent floor the radial, angular,
wall and origin closures cancel against each other, the eigenvalue error
changes sign with resolution and the measured order is noise: the
collocated ball Helmholtz error at m = 2, for instance, runs
`+5.7e-4, +2.7e-5, -4.6e-5, -3.5e-5, -1.3e-5` over
`N_r = 10, 12, 16, 18, 24`, crossing zero near `N_r = 14`.  The same effect
is documented for the sh_fd wall and origin stencils in
`docs/ball_sh_fd.md`.  So the tests

* assert the value at the fine resolution for every target;
* measure and print the order for every target;
* assert the order of the **RMS error over a benchmark's targets**, and only
  when the coarse RMS is above the floor
  (`3e-4` for fd2, `1e-5` for sh_fd, none for the staggered core, which is
  cleanly second order everywhere in this set).

## Radial eigenvalue conditions

All the Bessel conditions used here come from one helper.  With
`R(r) = a j_l(k r) + b y_l(k r)` (only `j_l` in a ball, by regularity) and
`x = k r`, the wall functionals are

| kind | condition | where it comes from |
|---|---|---|
| `dirichlet` | `f(x) = 0` | `s = 0`; toroidal no-slip `T = 0`; poloidal conducting `S = 0` |
| `neumann` | `f'(x) = 0` | `dR/dr = 0`; rigid wall on the acoustic pressure |
| `rT` | `f(x) + x f'(x) = 0` | `d(r T)/dr = 0`, the toroidal conducting wall |
| `T_over_r` | `x f'(x) - f(x) = 0` | `d(T/r)/dr = 0`, the toroidal stress-free wall |

and `k` solves `V_j(k r_i) V_y(k r_o) - V_j(k r_o) V_y(k r_i) = 0` in a
shell, `V_j(k r_o) = 0` in a ball.

## 1. Acoustic modes of a uniform gas sphere and shell

Linearised Euler about a uniform gas at rest, constant sound speed `c`,
`P = p1/rho0`:

    dt(v) = -Grad(P),    dt(P) = -c^2 Div(v).

For `e^{-i omega t}`, `v = Grad(P)/(i omega)` and the second equation
becomes `omega^2 P + c^2 Lap(P) = 0`, i.e. `Lap(P) = -(omega/c)^2 P`.  A
rigid wall `Dot(r_hat, v) = 0` is `dP/dr = 0`.  So `omega = c k` with

* ball: `j_l'(k) = 0` -- `k = 2.081576` (l = 1), `3.342094` (l = 2);
* shell `0.5 <= r <= 1`: the `neumann` determinant -- `k = 1.840268`
  (l = 1), `3.151176` (l = 2), `6.572013` (l = 0, the gravest purely radial
  p-mode; the half-wavelength estimate `pi/(r_o - r_i) = 6.283` is the
  plane-parallel check).

Parameters: `c = 2` on the ball (so the compile-time parameter is exercised
and every frequency doubles), `c = 1` on the shell; `m = 1` on the ball,
`m = 0` on the shell so that the radial (l = 0) mode exists.

The benchmark exercises Grad, Div and the pressure-velocity coupling with
no rotation, and nothing else: no Laplacian, no background.

## 2. Doppler shift under solid-body rotation

Let `U0 = rate * (z_hat x r) = rate * s e_phi`.  For a scalar,
`U0.Grad(s) = rate d/dphi s = i m rate s` exactly.  For a vector field, in
the local basis, `d/dphi e_i = z_hat x e_i`, so

    U0.Grad(v) = rate (d/dphi v + z_hat x v) = rate (i m v + z_hat x v),
    v.Grad(U0) = rate z_hat x v.

In a frame rotating at `Omega0 - rate` the fluid still rotates at `Omega0`
and carries the mean flow `U0`; the mean-flow and Coriolis terms are

    U0.Grad(v) + v.Grad(U0) + 2 (Omega0 - rate) z_hat x v
        = i m rate v + 2 Omega0 z_hat x v,

which is the inertial-mode operator of the co-rotating frame plus
`i m rate`.  Hence every eigenvalue satisfies

    omega(rate) = omega(0) + m * rate

with the eigenvector unchanged, **exactly** -- there is no discretisation
error in the shift.  The tests compare two nonzero rates,
`rate1 = 0.2` and `rate2 = 0.5`, so the predicted shift is
`m (rate2 - rate1)` and the advection operator is exercised in both
censuses (an identically zero `U0` would drop the term on every core).

Two problems are run:

* **scalar**: `dt(s) + U0.Grad(s) - kappa Lap(s) = 0` in a ball at m = 2
  with `s = 0` at `r = 1` and `kappa = 0.05`.  The spectrum is
  `lambda = kappa k^2 + i m rate` over `j_l(k) = 0`, so the decay rates are
  also asserted against the analytic `kappa k^2` (l = 2, 3).
* **vector**: the inertial-mode problem above at m = 1, with
  `v.Grad(U0) = rate z_hat x v` spelled as a Cross (an exact identity for a
  rigid-rotation background) and `U0.Grad(v)` left in its general form,
  which is what the benchmark tests.  Both the absolute frequency
  `omega(rate) - m rate = omega_i` (Zhang targets) and the shift are
  asserted.

The shift is asserted to `1e-10` absolute on every core; measured values
are `1e-16` to `3e-11` (see the table).

## 3. Inertial modes of a full sphere (Zhang et al. 2001)

`2 z_hat x v + Grad(p) = -dt(v)`, `Div(v) = 0`, `Dot(r_hat, v) = 0` at
r = 1: Poincare's problem, whose eigenfunctions in a full sphere are
polynomials in the cylindrical coordinates.  The curated `N = 1, 2` roots
of Zhang et al.'s (2001) eigenvalue equations 2.15 and 2.16 are reused from
`tests/test_evp_zhang_2001.py` (ten targets each at m = 1 and m = 2), with
the sign convention above.

**Poincare's operator is not compact**: its inviscid spectrum fills
`[-2, 2]` densely, and a discretisation fills the band with as many modes
as it has degrees of freedom.  A nearest match therefore proves nothing
unless it is much sharper than the mean spacing.  Each census is therefore
also matched against 40 random control frequencies drawn uniformly from
`[-1.9, 1.9]`, and the ratio

    discrimination = median control error / median target error

is reported and asserted.  A core that resolves individual inertial modes
has a large ratio; a core whose in-band spectrum is dense enough to fit any
frequency has a ratio of order 1.

On the collocated core at m = 2 the nearest neutral eigenvalue is not always
the mode: the non-adjoint Grad/Div pair lets a target collide with a band
neighbour into a complex pair at some resolutions, and a neutral gate then
returns the neighbour (Finding 4).  There every target is identified by its
eigenvector instead -- the mode within 0.05 of the target whose pressure
lies closest to the polynomial space of its Zhang class, the projection of
`tests/test_evp_zhang_2001.py` -- on the `18 x 27 -> 27 x 40` pair, where
that identification is unambiguous, and its real frequency is measured
whether or not the mode is neutral.

## 4. Free magnetic decay of a conducting shell

With `b = Curl(Curl(S r)) + Curl(T r)` both potentials obey
`dt(X) = eta Lap(X)`, so `X = a j_l(k r) + b y_l(k r)` and the rate is
`eta k^2`.

*Perfectly conducting wall.*  `b_r = l(l+1) S / r^2 = 0` gives `S = 0` at
each wall, which also kills the toroidal part of the wall electric field
`E = eta Curl(b)` (proportional to `Lap S = -k^2 S`); the poloidal part of
`E` is proportional to `d(r T)/dr`, so the toroidal family closes on
`d(r T)/dr = 0`.  In the shell `0.5 <= r <= 1` this gives the targets

| family | condition | k | rate k^2 |
|---|---|---|---|
| toroidal l = 1, n = 1 | `rT` | 1.984570 | 3.938519 |
| poloidal l = 1, n = 1 | `dirichlet` | 6.572013 | 43.191357 |
| poloidal l = 2, n = 1 | `dirichlet` | 7.111576 | 50.574517 |
| toroidal l = 1, n = 2 | `rT` | 6.618465 | 43.804079 |

which are the same four the staggered acceptance test uses.  In a ball the
conducting conditions become `j_l(k) = 0` (poloidal) and
`j_l(k) + k j_l'(k) = 0` (toroidal).

*Insulating exterior.*  Outside a ball of radius `a` the field is a
potential field decaying at infinity, so `S_ext ~ r^-(l+1)` and continuity
of `S, S'` gives `k j_l'(ka)/j_l(ka) = -(l+1)/a`.  With
`j_l' = j_{l-1} - (l+1) j_l / x` this collapses to

    j_{l-1}(k a) = 0        (poloidal, l = 1: k = pi, the classic dipole
                             free-decay rate pi^2)

while a toroidal field has no exterior at all, so `T(a) = 0`, i.e.

    j_l(k a) = 0            (toroidal).

The poloidal condition is a Robin condition **with an l-dependent
coefficient**.  The string BC language is local in (r, theta) and carries
no l, so it cannot express it on any core: on the collocated, staggered and
Dedalus cores it would mix harmonics, and although sh_fd works in the l
basis its BC strings have no way to name l.  The insulating case is
therefore not run as a poloidal benchmark.  Its toroidal specialisation,
`j_l(k a) = 0`, is exactly the no-slip toroidal condition of benchmark 7,
which is run on all four cores.

*Three spellings.*  The same physics is written three ways, which is where
the coverage matrix comes from:

* `b`: `dt(b) - eta Lap(b) = 0` **and** `Div(b) = 0` with no multiplier,
  plus `b_r = 0` and tangential `Curl(b) = 0`.  Only the staggered core
  reads the unmultiplied constraint as an involution (constrained
  transport); the other three see one more equation than unknowns.
* `A`: the vector potential with the Coulomb gauge as a saddle point --
  `dt(A) - eta Lap(A) - Grad(psi) = 0`, `Div(A) = 0`, tangential `A = 0`
  and `psi = 0` at the walls.  `b = Curl(A)` then satisfies `b_r = 0` and
  vanishing tangential `E`, so the rates are the same.
* `b_free`: `dt(b) - eta Lap(b) = 0` with the conducting wall alone and no
  constraint.  The solenoidal subspace is invariant (`Div Lap = Lap Div`),
  so the free-decay rates are in the spectrum.

`eta = 1`, `m = 0`, where the toroidal and poloidal families decouple.

## 5. Malkus field hydromagnetic inertial modes

Ideal MHD on the Malkus field `B0 = s e_phi = z_hat x r` in a rotating full
sphere, in units of `2 Omega` for frequency and with the Lehnert number
`Le` in `Le^2 (Curl(b) x B0)`:

    -i omega v + 2 z_hat x v = -Grad(p) + Le^2 (Curl(b) x B0),
    -i omega b = Curl(v x B0),    Div(v) = 0.

Two identities do everything.  For solenoidal `v` and `B0 = z_hat x r`,

    Curl(v x B0) = (B0.Grad)v - (v.Grad)B0
                 = (d/dphi v + z_hat x v) - z_hat x v = i m v,

so `b = -(m/omega) v` exactly -- the induced field is slaved to the flow
and is automatically divergence-free, which is why the problem needs
neither a `Div(b) = 0` row nor a wall condition on `b`.  And with
`Curl(B0) = 2 z_hat`,

    Curl(b) x B0 = -Grad(b.B0) + (b.Grad)B0 + (B0.Grad)b + b x Curl(B0)
                 = -Grad(b.B0) + z_hat x b + (i m b + z_hat x b) - 2 z_hat x b
                 = i m b - Grad(b.B0).

Substituting `b = -(m/omega) v` into the momentum equation gives

    -i (omega - Le^2 m^2 / omega) v + 2 z_hat x v = -Grad(P),

Poincare's operator again, with eigenvalue `omega_i = omega - Le^2 m^2 / omega`
and the **inertial eigenvector unchanged**.  Hence the exact dispersion
relation

    omega^2 - omega_i omega - Le^2 m^2 = 0,
    omega_pm = ( omega_i +- sqrt(omega_i^2 + 4 Le^2 m^2) ) / 2,

the fast branch `omega+ -> omega_i` and the slow (magneto-Coriolis) branch
`omega- -> -Le^2 m^2 / omega_i` as `Le -> 0`.  This is the classical
Malkus-field result (Malkus 1967); the derivation above is self-contained
and is what the test asserts -- no number is taken from the literature.
Two independent checks of the branch, both exact and both asserted on the
computed pair, are

    omega+ + omega- = omega_i,      omega+ omega- = -Le^2 m^2.

Run at m = 2, `Le = 0.1` and `0.3`, with `omega_i` taken from the exact
Zhang targets, so the prediction is closed form and does not borrow any
core's own inertial spectrum.  The wall condition is the impenetrable
`Dot(r_hat, v) = 0` of the hydrodynamic problem, which is also the
vanishing tangential electric field of a perfectly conducting sphere: the
tangential components of `v x B0` are `(-s v_r, 0)`.

Because the slow branch sits near zero, the Malkus errors are measured
against the frequency unit `2 Omega = 2` rather than against the target.
On the staggered core b is an involution (constrained transport, the
origin ring of the fluxes pinned, docs/ball_staggered.md section 7) and,
carrying no wall condition, keeps its wall fluxes as degrees of freedom
that evolve with the wall circulations of `Cross(v, B0)`, which vanish
on the Malkus field with `v_r = 0`; the product relation holds to `1e-6`
there because the discrete `Curl(Cross(v, B0))` is `i m v` on the C-grid
as well, and the errors are the inertial-mode errors themselves (RMS
order 2.2 to 2.3, tolerance `5e-3` from the largest 20x32 error, `2.8e-3`).

## 6. Gravity modes of a constant-N Boussinesq shell

    dt(v) = -Grad(p) + bu e_r,   dt(bu) = -N^2 v_r,   Div(v) = 0,

rigid walls `Dot(r_hat, v) = 0`.  For `e^{-i omega t}` and `p = P(r) Y_lm`,

    v_h = Grad_h(P)/(i omega),   v_r = alpha P'/(i omega),
    alpha = omega^2/(omega^2 - N^2),

and continuity `(r^2 v_r)'/r^2 + Div_h(v_h) = 0` becomes

    alpha (r^2 P')' = l(l+1) P,

an Euler equation with solutions `r^a`, `a(a+1) = l(l+1)/alpha` and
`a1 + a2 = -1`.  `P' = 0` at both walls needs
`(r_i/r_o)^(a1 - a2) = 1`, so `a1 - a2 = sqrt(1 + 4 l(l+1)/alpha)` is
imaginary, `= i q_n` with `q_n = 2 pi n / ln(r_o/r_i)`, `n = 1, 2, ...`
Then `1 + 4 l(l+1)/alpha = -q_n^2` and

    omega^2 = 4 l(l+1) N^2 / ( 1 + q_n^2 + 4 l(l+1) ).

*Independent check of the closed form.*  In a thin shell of thickness `d`
at radius `r`, `ln(r_o/r_i) -> d/r` and `q_n -> 2 k_r r` with
`k_r = n pi / d`, so the formula becomes

    omega^2 -> N^2 (l(l+1)/r^2) / ( l(l+1)/r^2 + k_r^2 )
             = N^2 k_h^2 / (k_h^2 + k_r^2),

the plane-parallel Boussinesq g-mode dispersion relation.

*Why a shell and not a ball.*  In a ball the same elimination gives
`a = -1/2 +- i q/2` with `q` real, so every solution oscillates without
bound as `r -> 0`: there is no discrete constant-N g-mode spectrum in a
full sphere, and the benchmark is only well posed on a shell.

Parameters: `N = 1`, `0.5 <= r <= 1`, `m = 1`; `q_1 = 9.0647`,
`q_2 = 18.1294`, giving `omega = 0.296225` (l = 1, n = 1), `0.473228`
(l = 2, n = 1) and `0.153920` (l = 1, n = 2).  The benchmark fixes the
relative sign of the buoyancy force `bu e_r` and the buoyancy production
`N^2 Dot(e_r, v)`: reversing either turns the oscillation into a growing
and a decaying pair, which the test would see as a missing frequency.

## 7. Diffusion sanity set on the shell

The ball half of this set is already spread over
`tests/test_ball_collocated.py`, `tests/test_dedalus_ball.py`,
`tests/test_sh_fd_ball.py` and `tests/test_ball_staggered.py`, each in its
own core's idiom; rather than duplicate it, the shell half is added here
and run on all four cores from one string problem.

* scalar `dt(s) = Lap(s)` at m = 0, 1, 2 with Dirichlet and with Neumann
  walls: rate `k^2` from the `dirichlet` / `neumann` determinants, l and
  l+1 for each m;
* toroidal viscous decay `dt(v) = Lap(v) - Grad(p)`, `Div(v) = 0` at
  m = 1, 2 with no-slip and impenetrable stress-free walls.  A toroidal
  field `v = Curl(T r)` is divergence-free and pressure-free and
  `Lap(v) = Curl(Lap(T) r)`, so it decays at `k^2` with `T = R(r) Y_lm`;
  no-slip is `R = 0` at both walls (the `dirichlet` determinant, and also
  the insulating-exterior condition on a decaying toroidal magnetic field),
  stress-free is `d(R/r)/dr = 0` (the `T_over_r` determinant).

## 8. Cross-core consistency

Three of the benchmarks -- the acoustic shell at m = 0, the constant-N
gravity shell at m = 1 and the Dirichlet scalar diffusion shell at m = 1 --
are read off all four cores at their finer resolution and the spread of the
four values is asserted below the tolerance of the coarsest core (the
staggered C-grid).  The spread is a stronger statement than each core's own
tolerance: a shared sign or convention error moves all four together, while
a discretisation error moves them apart.

## Coverage matrix

`ok` = compiled and benchmarked; everything else is the refusal the test
asserts.

| benchmark | fd2 | staggered | sh_fd | dedalus |
|---|---|---|---|---|
| 1 acoustic ball / shell | ok | ok | ok | ok |
| 2 Doppler, scalar | ok | ok | ok | ok |
| 2 Doppler, inertial | ok | ok | ok | ok |
| 3 inertial (Zhang) | ok | ok (no discrimination, see below) | ok | ok |
| 4 free decay, `b` (involution) | ok (divergence cleaning) | ok (constrained transport) | ok (divergence cleaning) | ok (divergence cleaning) |
| 4 free decay, `A` (saddle point) | ok | ok | ok | ok |
| 4 free decay, `b_free` (no constraint) | ok | tangential E on a non-flux unknown | ok | ok |
| 5 Malkus, ball | ok | ok | ok | ok |
| 6 gravity, constant-N shell | ok | ok | ok | ok |
| 7 diffusion set, shell | ok | ok | ok | ok |

## Measured accuracy

Max relative error over a benchmark's targets at each resolution, and the
order of the RMS error over them.  (Malkus is measured against the
frequency unit `2 Omega = 2`, not against the target, because its slow
branch sits near zero; Dedalus rows carry no order -- it is a spectral core
and both resolutions are converged.)

| benchmark | core | fine grid | targets | max err coarse | max err fine | rms order |
|---|---|---|---|---|---|---|
| acoustic ball m=1 | fd2 | 18x27 | 2 | 3.3e-04 | 5.8e-05 | 4.26 |
| acoustic ball m=1 | staggered | 20x32 | 2 | 2.9e-03 | 7.2e-04 | 1.99 |
| acoustic ball m=1 | sh_fd | 48x6 | 2 | 2.1e-06 | 1.3e-07 | 3.97 |
| acoustic ball m=1 | dedalus | 14x16 | 2 | 6.4e-16 | 2.1e-16 | -- |
| acoustic shell m=0 | fd2 | 18x27 | 3 | 5.2e-04 | 7.6e-05 | 4.78 |
| acoustic shell m=0 | staggered | 20x32 | 3 | 9.9e-03 | 2.5e-03 | 2.00 |
| acoustic shell m=0 | sh_fd | 48x6 | 3 | 5.4e-06 | 3.4e-07 | 3.48 |
| acoustic shell m=0 | dedalus | 14x16 | 3 | 2.7e-06 | 5.0e-09 | -- |
| doppler scalar decay m=2 | fd2 | 18x27 | 2 | 4.6e-04 | 3.5e-05 | 5.75 |
| doppler scalar decay m=2 | staggered | 20x32 | 2 | 9.9e-03 | 2.5e-03 | 2.01 |
| doppler scalar decay m=2 | sh_fd | 48x6 | 2 | 2.3e-05 | 2.6e-07 | 6.27 |
| doppler scalar decay m=2 | dedalus | 14x16 | 2 | 7.3e-16 | 2.7e-16 | -- |
| doppler inertial omega m=1 | fd2 | 18x27 | 3 | 1.7e-03 | 3.4e-04 | -- |
| doppler inertial omega m=1 | sh_fd | 48x6 | 3 | 8.1e-06 | 8.1e-06 | -- |
| doppler inertial omega m=1 | dedalus | 14x16 | 3 | 8.1e-06 | 8.1e-06 | -- |
| inertial zhang m=1 | fd2 | 18x27 | 10 | 6.2e-03 | 1.4e-03 | 3.59 |
| inertial zhang m=2 | fd2 | 27x40 | 10 | 5.2e-03 | 1.9e-03 | 3.41 |
| inertial zhang m=1 | staggered | 20x32 | 10 | 1.1e-02 | 5.3e-03 | 1.39 |
| inertial zhang m=2 | staggered | 20x32 | 10 | 1.8e-02 | 4.5e-03 | 2.37 |
| inertial zhang m=1 | sh_fd | 48x6 | 10 | 8.1e-06 | 1.0e-05 | -0.03 |
| inertial zhang m=2 | sh_fd | 48x6 | 10 | 5.7e-05 | 1.2e-05 | 2.25 |
| inertial zhang m=1 | dedalus | 14x16 | 10 | 1.0e-05 | 1.0e-05 | -- |
| inertial zhang m=2 | dedalus | 14x16 | 10 | 1.2e-05 | 1.2e-05 | -- |
| free decay A shell m=0 | fd2 | 18x27 | 4 | 2.7e-02 | 6.7e-03 | 3.46 |
| free decay b shell m=0 | staggered | 20x32 | 4 | 1.1e-02 | 2.7e-03 | 1.96 |
| free decay A shell m=0 | staggered | 20x32 | 4 | 9.5e-03 | 2.4e-03 | 1.81 |
| free decay b_free shell m=0 | sh_fd | 48x6 | 4 | 2.8e-04 | 2.1e-05 | 3.76 |
| free decay A shell m=0 | dedalus | 20x16 | 4 | 5.2e-04 | 7.9e-08 | -- |
| free decay b_free shell m=0 | dedalus | 20x16 | 4 | 2.5e-08 | 1.2e-12 | -- |
| malkus m=2 Le=0.1 | fd2 | 18x27 | 6 | 3.3e-02 | 1.9e-03 | -- |
| malkus m=2 Le=0.3 | fd2 | 18x27 | 6 | 1.1e-02 | 3.9e-03 | -- |
| malkus m=2 Le=0.1 | staggered | 20x32 | 6 | 1.0e-02 | 2.8e-03 | 2.28 |
| malkus m=2 Le=0.3 | staggered | 20x32 | 6 | 9.3e-03 | 2.5e-03 | 2.24 |
| malkus m=2 Le=0.1 | sh_fd | 48x6 | 6 | 6.2e-06 | 4.0e-06 | -- |
| malkus m=2 Le=0.3 | sh_fd | 48x6 | 6 | 4.9e-06 | 3.2e-06 | -- |
| malkus m=2 Le=0.1 | dedalus | 14x16 | 6 | 3.9e-06 | 3.9e-06 | -- |
| malkus m=2 Le=0.3 | dedalus | 14x16 | 6 | 3.5e-06 | 3.5e-06 | -- |
| gravity constant-N shell m=1 | fd2 | 18x27 | 3 | 3.1e-02 | 7.0e-03 | 3.64 |
| gravity constant-N shell m=1 | staggered | 20x32 | 3 | 1.1e-02 | 2.2e-03 | 2.48 |
| gravity constant-N shell m=1 | sh_fd | 48x6 | 3 | 4.5e-04 | 2.0e-06 | 7.23 |
| gravity constant-N shell m=1 | dedalus | 14x16 | 3 | 4.6e-06 | 1.2e-08 | -- |
| scalar dirichlet shell m=0 | fd2 | 18x27 | 2 | 5.7e-05 | 1.2e-05 | 3.71 |
| scalar neumann shell m=0 | fd2 | 18x27 | 2 | 5.6e-04 | 1.2e-04 | 3.71 |
| scalar dirichlet shell m=1 | fd2 | 18x27 | 2 | 5.3e-05 | 1.4e-05 | 3.38 |
| scalar neumann shell m=1 | fd2 | 18x27 | 2 | 5.6e-04 | 1.3e-04 | 3.67 |
| scalar dirichlet shell m=2 | fd2 | 18x27 | 2 | 8.9e-05 | 2.1e-05 | 3.44 |
| scalar neumann shell m=2 | fd2 | 18x27 | 2 | 6.3e-04 | 1.3e-04 | 3.75 |
| scalar dirichlet shell m=0 | staggered | 20x32 | 2 | 9.5e-03 | 2.4e-03 | 1.99 |
| scalar neumann shell m=0 | staggered | 20x32 | 2 | 8.5e-03 | 2.1e-03 | 2.00 |
| scalar dirichlet shell m=1 | staggered | 20x32 | 2 | 8.9e-03 | 2.2e-03 | 1.99 |
| scalar neumann shell m=1 | staggered | 20x32 | 2 | 8.8e-03 | 2.2e-03 | 1.99 |
| scalar dirichlet shell m=2 | staggered | 20x32 | 2 | 8.9e-03 | 2.2e-03 | 1.99 |
| scalar neumann shell m=2 | staggered | 20x32 | 2 | 9.3e-03 | 2.3e-03 | 2.01 |
| scalar dirichlet shell m=0 | sh_fd | 48x6 | 2 | 1.3e-06 | 1.5e-07 | 2.85 |
| scalar neumann shell m=0 | sh_fd | 48x6 | 2 | 6.3e-05 | 4.7e-06 | 3.72 |
| scalar dirichlet shell m=1 | sh_fd | 48x6 | 2 | 2.3e-06 | 1.1e-07 | 4.37 |
| scalar neumann shell m=1 | sh_fd | 48x6 | 2 | 6.3e-05 | 4.7e-06 | 3.73 |
| scalar dirichlet shell m=2 | sh_fd | 48x6 | 2 | 4.3e-06 | 3.8e-08 | 6.71 |
| scalar neumann shell m=2 | sh_fd | 48x6 | 2 | 6.1e-05 | 4.6e-06 | 3.71 |
| scalar dirichlet shell m=0 | dedalus | 14x16 | 2 | 3.1e-08 | 2.1e-11 | -- |
| scalar neumann shell m=0 | dedalus | 14x16 | 2 | 5.2e-06 | 8.6e-09 | -- |
| scalar dirichlet shell m=1 | dedalus | 14x16 | 2 | 9.1e-08 | 8.3e-11 | -- |
| scalar neumann shell m=1 | dedalus | 14x16 | 2 | 7.3e-06 | 1.6e-08 | -- |
| scalar dirichlet shell m=2 | dedalus | 14x16 | 2 | 1.8e-07 | 2.1e-10 | -- |
| scalar neumann shell m=2 | dedalus | 14x16 | 2 | 9.5e-06 | 2.7e-08 | -- |
| toroidal noslip shell m=1 | fd2 | 18x27 | 2 | 1.1e-04 | 2.0e-05 | 4.01 |
| toroidal stress_free shell m=1 | fd2 | 18x27 | 2 | 7.0e-04 | 1.1e-04 | 4.51 |
| toroidal noslip shell m=2 | fd2 | 18x27 | 2 | 6.8e-05 | 2.2e-05 | 2.85 |
| toroidal stress_free shell m=2 | fd2 | 18x27 | 2 | 4.0e-04 | 9.1e-05 | 3.65 |
| toroidal noslip shell m=1 | staggered | 20x32 | 2 | 5.9e-03 | 1.4e-03 | 2.02 |
| toroidal stress_free shell m=1 | staggered | 20x32 | 2 | 7.5e-03 | 1.9e-03 | 1.99 |
| toroidal noslip shell m=2 | staggered | 20x32 | 2 | 4.2e-03 | 9.7e-04 | 2.09 |
| toroidal stress_free shell m=2 | staggered | 20x32 | 2 | 7.3e-03 | 1.9e-03 | 1.91 |
| toroidal noslip shell m=1 | sh_fd | 48x6 | 2 | 2.3e-06 | 1.1e-07 | 4.37 |
| toroidal stress_free shell m=1 | sh_fd | 48x6 | 2 | 2.4e-05 | 1.8e-06 | 3.42 |
| toroidal noslip shell m=2 | sh_fd | 48x6 | 2 | 4.3e-06 | 3.8e-08 | 6.71 |
| toroidal stress_free shell m=2 | sh_fd | 48x6 | 2 | 2.8e-05 | 2.0e-06 | 3.76 |
| toroidal noslip shell m=1 | dedalus | 14x16 | 2 | 9.1e-08 | 8.3e-11 | -- |
| toroidal stress_free shell m=1 | dedalus | 14x16 | 2 | 4.2e-06 | 9.2e-09 | -- |
| toroidal noslip shell m=2 | dedalus | 14x16 | 2 | 1.8e-07 | 2.1e-10 | -- |
| toroidal stress_free shell m=2 | dedalus | 14x16 | 2 | 5.9e-06 | 1.7e-08 | -- |

Tolerances asserted at the fine resolution, chosen roughly 2-4x above the
measured error of the worst target so that the test is a regression guard
rather than a coincidence:

| benchmark | fd2 | staggered | sh_fd | dedalus |
|---|---|---|---|---|
| acoustic | 3e-4 | 4e-3 | 2e-5 | 1e-8 |
| scalar diffusion | 2e-4 | 4e-3 | 5e-5 | 1e-7 |
| toroidal diffusion | 3e-4 | 8e-3 | 1e-4 | 1e-7 |
| free decay | 1e-2 | 1.5e-2 | 5e-4 | 1e-6 |
| inertial (Zhang) | 8e-3 | 2e-2 | 1e-4 | 1e-4 |
| Malkus (absolute) | 5e-3 | 5e-3 | 5e-5 | 5e-5 |
| gravity | 1e-2 | 1e-2 | 2e-4 | 1e-7 |
| Doppler shift (absolute) | 1e-10 | 1e-10 | 1e-10 | 1e-10 |

Three tolerances are set by something other than the discretisation and are
worth naming: the Zhang targets are tabulated to five decimals, so the
sh_fd and Dedalus inertial and Malkus errors bottom out near `1e-5`
relative and `4e-6` absolute; the Dedalus shell rows are limited by the
radial resolution rather than by rounding (`5e-9` at 14 radial modes,
`8e-11` at 20); and the collocated gravity tolerance is set by the n = 2
g-mode, which carries three radial wavelengths in `0.5 <= r <= 1` and is
only marginally resolved at `N_r = 18` (error `7.0e-3`, converging at order
3.6).

## Doppler shift: the exact statement

Every core reproduces `omega(rate2) - omega(rate1) = m (rate2 - rate1)` to
rounding.  Worst measured absolute error over both resolutions and all
targets:

| problem | fd2 | staggered | sh_fd | dedalus |
|---|---|---|---|---|
| scalar, m = 2, shift 0.6 | 4.9e-15 | 1.3e-15 | 1.5e-15 | 5.6e-16 |
| inertial, m = 1, shift 0.3 | 2.7e-11 | 8e-15 | 2.3e-14 | 1.7e-14 |

The collocated inertial figure (`1e-14` to `3e-11`) is the conditioning of
the saddle-point pencil, not a discretisation error: the shift is exact for
every mode of the census, including the spurious ones.

## Inertial modes: discrimination against a dense band

Median nearest-match error of the ten curated targets, of 40 random control
frequencies in `[-1.9, 1.9]`, and the number of neutral modes the census
puts in `|omega| < 2`.

| core | grid | modes in band | target | control | ratio |
|---|---|---|---|---|---|
| fd2 | 12x18 | 201 | 7.1e-04 | 7.1e-03 | 10.0 |
| fd2 | 18x27 | 408 | 2.6e-04 | 3.6e-03 | 13.6 |
| fd2 (m = 2) | 18x27 | 381 | 7.7e-04 | 4.2e-03 | 5.5 |
| fd2 (m = 2) | 27x40 | 814 | 3.2e-04 | 2.6e-03 | 8.2 |
| staggered | 10x16 | 288 | 4.5e-03 | 2.0e-03 | 0.4 |
| staggered | 20x32 | 1212 | 1.5e-03 | 8.4e-04 | 0.6 |
| staggered (m = 2) | 20x32 | 1214 | 1.1e-03 | 9.7e-04 | 0.9 |
| sh_fd | 48x6 | 657 | 3.6e-06 | 1.2e-02 | 3259 |
| dedalus | 14x16 | 293 | 3.5e-06 | 1.2e-02 | 3345 |

The staggered band counts are 2 N_r lower than they were before the pole
rings were pinned explicitly (308 -> 288 and 1252 -> 1212): those were the
spurious `lambda = 0` eigenvalues of the accidental `0 = lambda x` pin
(docs/staggered_general_design.md section 3b).  No target or control
distance moved.

## Cross-core consistency

Fine resolution of each core; spread is the largest deviation from the
four-core mean, relative to the analytic value.

| benchmark | analytic | fd2 | staggered | sh_fd | dedalus | spread |
|---|---|---|---|---|---|---|
| acoustic shell m=0 | 1.84026818 | 1.84025662 | 1.83852368 | 1.84026871 | 1.84026818 | 7.1e-04 |
| acoustic shell m=0 | 3.15117606 | 3.15093795 | 3.14335866 | 3.15117694 | 3.15117607 | 1.8e-03 |
| acoustic shell m=0 | 6.57201320 | 6.57190595 | 6.56501375 | 6.57201095 | 6.57201320 | 8.0e-04 |
| gravity shell m=1 | 0.29622454 | 0.29620744 | 0.29556246 | 0.29622404 | 0.29622454 | 1.7e-03 |
| gravity shell m=1 | 0.47322831 | 0.47318427 | 0.47246872 | 0.47322763 | 0.47322831 | 1.2e-03 |
| scalar dirichlet m=1 | 43.19135749 | 43.19085419 | 43.09818208 | 43.19135292 | 43.19135749 | 1.6e-03 |
| scalar dirichlet m=1 | 50.57451659 | 50.57381097 | 50.46116314 | 50.57451467 | 50.57451659 | 1.7e-03 |

In every row the spread is set by the staggered core's second-order error
and the other three agree with each other, and with the analytic value, far
more tightly.  No sign or convention disagreement between cores appears
anywhere in the set.

## Findings

1. **The staggered core does not discriminate individual inertial modes.**
   Its in-band inviscid spectrum is dense enough that 40 random control
   frequencies are matched as well as the ten curated Zhang targets (ratio
   0.4 to 1.1, table above): recovering a target is a property of the mode
   density, not of the mode.  The other three cores do discriminate:
   5-14x for the collocated core and ~3300x for sh_fd and Dedalus.

   This applied to `tests/test_ball_staggered.py::TestZhangBall` as well,
   which asserted a max relative error of 1.5e-2 at 16x32 and 6e-3 at
   24x48 -- at its own resolutions the same measurement gives 960 and 2206
   modes in the band and a target/control ratio of 0.83 and 1.64 (m = 1),
   1.00 and 1.08 (m = 2), so those thresholds sat at or below the mean
   spacing.  The convergence of its RMS error is real, so that class now
   asserts the RMS over the target set (2.20e-3 -> 7.72e-4 at m = 1, order
   2.58; 3.12e-3 -> 7.11e-4 at m = 2, order 3.65) and reports the ratio
   without asserting it, and
   `tests/test_evp_zhang_2001.py::TestZhang2001StaggeredInertialWaves`
   keeps only the band-coverage claim (a mode within 0.03 of every target,
   the guard against a half-strength Coriolis).

2. **An identically zero background vector crashed the Dedalus ball path.**
   RESOLVED.  `add_background("U0", v_r=0, v_theta=0, v_phi="0.0*r*sin(theta)")`
   compiled on a shell, and with any nonzero coefficient on a ball, but on a
   ball with every component zero the NCC branch of
   `spdee/backends/dedalus/compiler.py::_product` translated the background
   to the Python float `0.0` and evaluated `0.0 @ Grad(s)`, raising
   `TypeError`.  The product now drops a term with a numeric zero factor, as
   Dedalus's own Multiply does, for every rank and product position
   (`tests/test_dedalus_ball.py::test_zero_background_drops_the_term`).  The
   Doppler benchmark keeps comparing two nonzero rates so that the advection
   operator is present in both censuses.

3. **The collocated ball Helmholtz error changes sign near N_r = 14.**
   At m = 2 the relative eigenvalue error runs
   `+5.7e-4, +2.7e-5, -4.6e-5, -3.5e-5, -1.3e-5` over
   `N_r = 10, 12, 16, 18, 24` (N_theta = 1.5 N_r), so a resolution pair
   straddling the crossing measures an order of -0.6 while the value itself
   is correct to 3e-5.  This is the same effect `docs/ball_sh_fd.md`
   records for the sh_fd wall and origin stencils, and it is why the tests
   assert an RMS order above a floor rather than a per-target order.

4. **At m = 2 the collocated core's nearest neutral eigenvalue is not
   always the Zhang mode: mode collision.**  On the 12x18 / 18x27 censuses
   three targets (`omega_Z = 1.64344, 1.15306, 1.73397`) showed orders
   0.07, -0.50 and -1.07 with errors at the local mode spacing (4.2e-3).
   Identifying the modes by eigenvector
   (`tests/test_evp_zhang_2001.py::test_m2_slow_targets_identified_by_pressure_polynomial`)
   shows why: at those resolutions each of the three has collided with a
   neighbouring band mode into a complex pair with `|Im omega|` of 1e-3 to
   1e-2 (the non-adjoint collocated Grad/Div), the neutral gate drops the
   pair, and the nearest neutral survivor is the neighbour, whose distance
   does not fall with N.  The real parts of the pairs do converge (2e-4 to
   4e-4 at 24x48, order 4) and the pairs close with resolution.  The
   benchmark now identifies every m = 2 target by its pressure polynomial on
   the 18x27 / 27x40 pair: the three converge at 3.1e-3 -> 3.1e-4,
   5.2e-3 -> 2.9e-4 and 5.2e-3 -> 2.5e-4, and the RMS order over the ten is
   3.41.  The collision moves with resolution -- at 27x40 it is the N = 1
   SYM target `omega_Z = 1.23193` that sits in a pair
   (`-1.23425 -+ 0.00118 i`, error 1.9e-3 against 5.0e-4 at 18x27), which
   is the largest fine-grid error of the set.  The polynomial residuals of
   the identified eigenvectors are 2e-4 to 0.16 at 27x40 (order 4), against
   below 2e-2 at 24x48 with order-6 stencils: the collocated eigenvectors
   carry a non-polynomial pressure component at these resolutions, so the
   identification is by the best fit within a window rather than by a
   clean membership test.  The m = 1 set converges cleanly (RMS order 3.59)
   without identification.

5. **The vector-potential formulation of free decay converges more slowly
   than the `b` formulation on the staggered core** for the poloidal l = 1
   mode (order 1.07 against 1.87 for the same target), although both reach
   the same fine-grid accuracy and the RMS orders are 1.81 and 1.96.  Not
   chased further here.

6. **One refusal is by design.**  A written `Div(b) = 0` without a
   multiplier is an involution on the staggered core (constrained transport,
   the row dropped) and a divergence-cleaning saddle point with an
   auto-introduced multiplier on the other three, so the `b` formulation runs
   everywhere.  The `b_free` formulation, which drops the constraint and
   closes the vector Laplacian with the conducting wall alone, is refused by
   the staggered core, which reads a tangential electric field only on a flux
   unknown; the other three build it from the tangential `Curl(b)` rows.  The
   Doppler benchmark runs on every core: the staggered contraction with a
   theta-dependent background equals its Coriolis block entrywise for a
   rigid-rotation background, and the shift is exact to rounding there too.

## Runtime

The whole module is 91 test items and about 10 minutes on three threads.
The Dedalus compiles dominate (one subproblem per l), the dense censuses
are 1-40 s each.  Nothing is marked `slow`, since no single item exceeds
30 s, but the module is heavier than the rest of the default suite.
