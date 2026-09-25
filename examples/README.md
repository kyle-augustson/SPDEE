# SPDEE examples

Five scripts for readers who know the physics and want to see how a problem is
stated, run on a core, checked against a closed form, and polished with the
solver layer.  Each is a plain script of functions with a `main()`, prints a
compact table and saves one figure to `examples/figures/`.  They are meant to
be read; they also run, in about a minute each at their default resolution.

## Running

From the repository root, in the environment that carries SPDEE and its cores:

    PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python examples/01_helmholtz.py

The Dedalus core needs Dedalus 3; `spdee.solvers.available()` says which
optional solver components (FPLUS, bbjd, CuPy, cuDSS) are present, and every
example falls back to SuperLU and host LAPACK without them.  Run one example at
a time.  `tests/test_examples.py` imports each example's functions and runs a
reduced-resolution smoke of every one (about a minute in all).

## The examples

**01_helmholtz.py** — scalar Helmholtz eigenvalues of `dt(u) = lap(u)` in a
ball and a shell with Dirichlet, Neumann and Robin walls, one `Problem` compiled
on all four cores at three resolutions, every eigenvalue against the spherical
Bessel roots of the wall condition.  Shows the measured order of each core
(4 for the collocated and sh_fd stencils used, 2 for the C-grid, spectral for
Dedalus), the full census with `dense`, the eigenvalue convention, and how each
core hands an eigenvector back onto a meridional grid.

**02_inertial_modes.py** — inviscid inertial modes of a rotating full sphere on
the sh_fd core against the Zhang et al. (2001) frequencies at m = 1, 2; the
exact Doppler shift of every eigenvalue by `m (rate2 - rate1)` under a change
of rotating frame, to rounding, as the sharpest check of the advection operator
against the Coriolis operator.  Shows a saddle-point constraint with the
pressure gauge, why a dense band needs eigenvector-based identification (the
candidates within a window ranked by the fit of their pressure to the Zhang
polynomial space), shift-invert with `v0` seeding, and a meridional plot.

**03_convective_onset.py** — Boussinesq convection in a rotating shell (Ek, Pr,
Ra in the rotational scaling, stress-free or no-slip walls, fixed temperature):
the critical Rayleigh number at fixed m from a growth-rate scan bracketed with
`brentq`, on the Dedalus core and on the staggered C-grid at two resolutions,
with the marginal mode's temperature from each.  Shows a symbolic parameter
bound at compile time (`add_parameter("Ra")`, `params={"Ra": value}`), that a
linear parameter needs two assemblies for a whole scan (`K(Ra) = K0 + Ra K1`),
and second-order cross-core agreement against the spectral value.

**04_critical_layer.py** — the sectoral Rossby mode of a shell with the
latitudinal differential rotation `Omega(theta) = Omega0 (1 - eps cos^2 theta)`,
entered as a background flow with its advection `dot(U0, grad(v))`, the
contraction `dot(v, gradU0)` with the hand-derived gradient tensor of the flow,
and the Coriolis term of the frame.  At finite Ek the eigenfunction develops a
critical layer where `omega = m (Omega(theta_c) - Omega0)`.  This is the
polishing workflow: coarse ARPACK to locate, finer ARPACK from the coarse
eigenvalue, bbjd from the ARPACK vector and from the coarse eigenvector
interpolated onto the fine grid (residual 7e-2 to 2e-13 in seconds), a dense
GPU census of the coarse pencil as the ground truth of the window, and
convergence of the eigenvalue and of the layer width with resolution and Ek.

**05_magnetic_decay.py** — free decay of a magnetic field in a perfectly
conducting shell written three ways (the field with `div(b) = 0`, the vector
potential in the Coulomb gauge, the field closed by the conducting wall alone),
on every core, against the Bessel conditions of the toroidal and poloidal
families.  Shows the constraint classifier (`spdee.ir.constraints.classify`:
involution kept by constrained transport on the C-grid and by divergence
cleaning elsewhere, the saddle point of the Coulomb gauge, the refusal of the
free spelling by the C-grid), the divergence of the eigenvectors per core, and
that all spellings converge to the same rates while the free spelling also
carries the non-solenoidal gradient modes.

**06_conditioning_and_nonnormality.py** — the tools of `spdee.analysis` on
two pencils.  The collocated inertial ball: modes ranked by eigenvalue
condition number, the sectoral r-mode at 1.1 and the checkerboard direction
with a singular value at rounding, which is deflated and the pencil re-solved
to neutral modes.  A shell with a prescribed differential rotation, a shear
flow and therefore non-normal: the pseudospectrum around the least-damped mode,
the numerical against the spectral abscissa, the resolvent norm along the
imaginary axis against the normal bound, continuation of the mode in the shear
strength, and a Jacobi-Davidson polish with the two-grid preconditioner.

## The common workflow

1. **State** the problem once: `pb = Problem()`, `add_unknown`,
   `add_background` (sympy expressions in `r`, `theta`, or tables),
   `add_parameter` (numeric, or symbolic and bound at compile time),
   `add_equation("lhs = rhs")` in the Dedalus spelling, `add_bc(row, at=wall)`
   or `add_stress_free(vector, at=wall)`.
2. **Compile** on a core at one azimuthal order m:
   `pb.compile(grid, m, backend="fd2"|"sh_fd"|"dedalus", gauge_unknowns=[...],
   params={...})`, with `discretization="staggered"` on `fd2` for the C-grid.
   Every compiled problem gives the pencil `K, M = cp.assemble_evp_matrices()`.
3. **Solve** with `spdee.solvers`: `shift_invert(K, M, sigma, k, ...)` for the
   k eigenpairs nearest a shift, `dense(K, M, sigma, backend=...)` for the whole
   spectrum of a small pencil.
4. **Check the residual** of every eigenpair you keep,
   `||K x - lambda M x|| / (||K x|| + |lambda| ||M x||)` (`spdee.io.pencil_residuals`),
   and for constrained systems gate the census on it.
5. **Converge in resolution**: repeat at two or three resolutions and measure
   the order; a value that does not move with the grid is the answer.
6. **Cross-check on a second core**: a discretisation error moves the cores
   apart, a sign or convention error moves them together.

## The eigenvalue convention

Every core assembles `K x = lambda M x` with the `dt` terms in M and the rest
in K.  Modes go as `e^{i m phi} e^{lambda t}` and the physical frequency is

    omega = -i lambda        Re omega = Im lambda   (oscillation, retrograde < 0)
                             Im omega = -Re lambda  (growth; damped < 0)

A decay rate gamma > 0 is `lambda = gamma` (omega = -i gamma); a neutral wave of
frequency omega is `lambda = i omega`.  A shift near a target frequency
`omega_t` is therefore `sigma = 1j * omega_t + offset`, the offset a small
positive real number placing the shift on the damped side of the target.

## Which solver

| need | call |
|---|---|
| a few eigenpairs near a target on a large pencil | `shift_invert(K, M, sigma, k)` (ARPACK over SuperLU, the default) |
| the same, faster factorization | `factor="cudss"` (GPU; it gates itself on a residual check and a nearly singular `K - sigma M` fails it, so keep the shift a little off the eigenvalue), `factor="cupy"`, `factor="auto"` (cudss, cupy, then scipy) |
| a spectral core's pencil whose fill is dense anyway | `factor="lapack"`, or `factor="fplus"` (int64 LU with selectable pivoting, built for those pencils; far slower than SuperLU on 2-D finite-difference pencils) |
| polish one eigenpair to rounding from a good start | `method="bbjd"` with `v0=` the starting vector (ARPACK's, or a coarse eigenvector interpolated to the grid); stalls on pencils with a dense integral gauge row (the m = 0 pressure gauge) |
| the whole spectrum of a pencil up to a few thousand unknowns | `dense(K, M, sigma, backend="auto")` (cuSOLVER Xgeev on the GPU, LAPACK on the host); the ground truth for what a k-nearest solve misses |
| a scan in a parameter that enters linearly | assemble twice and combine, `K(p) = K0 + p K1` (example 03) |
