# Meridional (θ, r) non-constant coefficients on the Dedalus-3 ball

Brief for the Dedalus developers. Prepared against Dedalus 3.0.5, upstream `master` at commit
`2a2ad151`, editable install from upstream master; all `file:line` references are to that
tree. The accompanying script `dedalus_ball_meridional_ncc_prototype.py` (same directory) contains the
prototype matrix builder, the verification, an end-to-end LBVP, the nnz counts, and the four patches
described in section 6 in executable (monkeypatch) form.

## 0. Summary

* **What blocks it.** A `(θ, r)` NCC on `BallBasis` is *not* rejected at the domain-arithmetic level:
  `BallBasis.__rmatmul__` (`dedalus/core/basis.py:4825`) accepts any `BallBasis` NCC, `Multiply.matrix_coupling`
  (`dedalus/core/arithmetic.py:540`) correctly infers ℓ-coupling from `ncc.domain.nonconstant`, and `build_solver`
  builds subproblems with groups `(m, None, None)`. It fails when the NCC matrix is requested:
  `BallBasis` has no `build_ncc_matrix`, so it inherits `Basis.build_ncc_matrix` (`basis.py:272-282`), which raises
  `NotImplementedError("Only last-axis NCCs implemented for this basis.")` at **`basis.py:276`**. `ShellBasis`
  overrides it (`basis.py:4667-4681`) and dispatches to `ShellBasis.build_meridional_ncc_matrix` (`4683-4722`);
  there is also no `BallBasis.meridional_basis` (the shell's is at `basis.py:4626-4630`). The tests carry the
  markers `xfail(reason="Ball meridional NCCs not implemented.")` at `dedalus/tests/test_spherical_ncc.py:160,193,226`.
* **Dense or banded.** Banded. For one NCC mode `a_{l'p}(r) Y_{l'}^0` of polynomial degree `d_r = 2p + l'` in `r`,
  every radial block `(ℓ → ℓ'')` has exactly `d_r + 1` diagonals (plus `dk` for the `k`-conversion), and the
  angular coupling is `|ℓ − ℓ''| ≤ l'` with `ℓ + l' + ℓ''` even. The blocks *differ* from one another (they
  depend on `(l', ℓ, ℓ'')`), which is what defeats the shell's Kronecker-Clenshaw shortcut; nothing becomes
  dense. The recollection "dense because of n–ℓ coupling" should be replaced by "block-varying but banded, with
  the same bandwidth law as the shell".
* **Numbers** (m = 0 pencil, `N_r = 32`, `l_max = 63`, NCC with `l' ≤ 4` and radial degree `≤ 8`):
  ball meridional **64.5 nnz/row mean, 77 max** (68,138 nnz over 1056 rows, 6.1 % of dense);
  shell meridional, Dedalus's own matrix, 128.1 / 153 (262,432 nnz over 2048 rows); ball radial-only,
  Dedalus's own, 7.8 / 9. The meridional-over-radial factor is `2L' + 1 = 9` on the ball, exactly as on the
  shell; the ball's blocks are half the shell's width because the Zernike index counts `r²`.
* **Verification.** The prototype reproduces Dedalus's grid-space product to `1e-15` relative in seven
  configurations (radii 1–2; `k_ncc, k_arg ∈ {0,1,2}`; `dk ∈ {0,1,2}`; resolutions (8,8,8) and (8,12,12)),
  and, with four small patches to the ball's ℓ-coupled machinery, the LBVP
  `-lap(u) + (1 + ½ r cos θ) u + lift(τ) = F, u(r=1) = 0` solves through `d3.LBVP`.
* **Three further gaps** in the ball's ℓ-coupled path, never exercised because no ball operator couples ℓ:
  (i) `SphericalEllOperator._coupled_ell_matrices` (`dedalus/core/operators.py:3225-3233`) builds the compact
  `Σ_ℓ n_size(ℓ)` layout while ℓ-coupled ball subproblems use the padded `(n_ℓ, N_max + 1 − ⌊|m|/2⌋)` layout;
  (ii) `LiftBall.subproblem_matrix` (`basis.py:5215-5247`) assumes a separable ℓ (`LiftShell`, `5328-5347`, does
  not); (iii) `Spherical3DBasis.valid_elements` (`basis.py:4440-4463`) does not invalidate ball slots with
  `n < ⌊ℓ/2⌋` — in ℓ-separable subproblems they are excluded implicitly by `n_slice(ℓ)`, in ℓ-coupled ones they
  survive as empty rows/columns and the LU is exactly singular.

## 1. Where it stops today

| Step | Location | Behaviour |
|---|---|---|
| NCC basis | none | No `BallBasis.meridional_basis`; `b.clone_with(shape=(1,) + b.shape[1:])` works as the NCC basis (transforms, products in grid space all fine). |
| Domain arithmetic | `basis.py:4825-4833` (`BallBasis.__rmatmul__`) | Accepts; output `k = max(k_ncc, k_arg)` via `BallRadialBasis.__rmatmul__`. (`BallBasis.__mul__`, `4811-4820`, gives `k = 0` — pre-existing asymmetry.) |
| Coupling | `arithmetic.py:540-545` | `ncc.domain.nonconstant = (F, T, T)` ⇒ `matrix_coupling = [False, True, True]`, subproblems `(m, None, None)`. Verified. |
| NCC matrix | `arithmetic.py:407-416` → `basis.py:272-276` | **`NotImplementedError("Only last-axis NCCs implemented for this basis.")`** |
| Second guard (if the first were bypassed) | `basis.py:4481-4490` (`Spherical3DBasis._last_axis_component_ncc_matrix`) | `ValueError("Cannot build NCCs of non-radial fields.")` |
| Radial NCC inside an ℓ-coupled ball subproblem | `basis.py:4247` (`ell = subproblem.group[1]  # HACK`) | `group[1]` is `None`; the shell's counterpart is ℓ-independent (`ell = 0  # HACK`, `basis.py:4037`) so a shell problem may mix radial and meridional NCCs (`tests/test_lbvp.py:545`), the ball cannot. |
| Other ℓ-coupled operators | `operators.py:3225`, `basis.py:5215`, `basis.py:4440` | Layout mismatch / separable-ℓ assumption / undropped invalid slots, see section 6. |

## 2. How the shell does it, and why it does not transfer

Shell (`ShellBasis.build_meridional_ncc_matrix`, `basis.py:4683-4722`): the NCC is `a(r,θ) = Σ_n a_n(θ) T_n(r)`
with an ℓ-independent radial basis. The code converts the NCC to spin components, builds for each radial index
`n` the S2 NCC matrix `M_n` of `a_n(θ)` (Gaunt coupling over ℓ, computed by a Clenshaw recurrence in `cos θ` on
the argument's spin-weighted Jacobi matrix, `SphereBasis._last_axis_component_ncc_matrix`, `basis.py:3394-3433`),
conjugates by the regularity intertwiners `Q_ℓ` (`4708-4717`), and hands the list `{M_n}` to
`ShellRadialBasis._last_axis_component_ncc_matrix` (`4036-4072`) as *matrix-valued coefficients*:
`clenshaw.kronecker_clenshaw` (`dedalus/tools/clenshaw.py:41-63`) forms `Σ_n kron(M_n, T_n(J_r))`, with `J_r` the
single Jacobi matrix of the radial basis (`ell = 0  # HACK`, line 4037). This works only because multiplication
by a function of `r` is the same operator for every ℓ on the shell.

Ball: the radial basis is `Q_n^{(k,ℓ)}(r) = c_{kℓ} r^ℓ P_n^{(α+k, ℓ+½)}(2r² − 1)`
(`dedalus/libraries/dedalus_sphere/zernike.py:29-43`; with `regtotal`, `ℓ → ℓ + regtotal`,
`basis.py:4172-4198`), so the Jacobi matrix `J_ℓ`, the `r^{l'}` prefactor, and the target family all depend on
`(ℓ, l', ℓ'')`. There is no Kronecker factorisation, so the per-ℓ blocks must be built one by one. That is the
"n–ℓ coupling"; it changes the *assembly*, not the sparsity.

## 3. Mathematics

**Bases.** Field: `u = Σ_{ℓ,m,n} u_{ℓmn} Q_n^{(k,ℓ)}(r) Y_ℓ^m`, `Q_n^{(k,ℓ)} ∝ r^ℓ P_n^{(α+k, ℓ+½)}(z)`,
`z = 2r² − 1`, `r` normalised by the ball radius. `Q_n^{(k,ℓ)}` is a polynomial of total Cartesian degree `2n + ℓ`.
Dedalus stores `n = 0 … N_max − ⌊ℓ/2⌋` for each ℓ (`BallRadialBasis._nmin`, `basis.py:4242`; `n_size`, `3827`;
`BallRadialTransform`, `dedalus/core/transforms.py:1451-1500`): a triangular truncation at total degree
`≤ 2N_max + 1`. In the coefficient array the Zernike index `n` sits at position `n + ⌊ℓ/2⌋`; an ℓ-coupled
subsystem's radial slice starts at `⌊|m|/2⌋` (`Subsystem.coeff_slices`, `dedalus/core/subsystems.py:129-146`).

**NCC.** Axisymmetric scalar: `a(r,θ) = Σ_{l'} a_{l'}(r) Y_{l'}^0`, and the ball transform of a smooth field
automatically gives `a_{l'}(r) = Σ_p c_{l'p} Q_p^{(k_a, l')}(r) = r^{l'} × (polynomial in r² of degree ≤ p_max)`.
Regularity at the origin is built into the NCC's own expansion; no extra condition is needed.

**Angular factor.** `Y_{l'}^0 Y_ℓ^m = Σ_{ℓ''} G^m_{ℓ l' ℓ''} Y_{ℓ''}^m`, Gaunt coefficients with
`|ℓ − l'| ≤ ℓ'' ≤ ℓ + l'` and `ℓ + l' + ℓ''` even. Dedalus's S2 Clenshaw already produces `G` in its own SWSH
normalisation; the prototype obtains the matrix of a single `Y_{l'}^0` by feeding a unit coefficient vector
through `SphereBasis._last_axis_component_ncc_matrix` and asserts the selection rules.

**Radial factor.** `a_{l'}(r) Q_n^{(k,ℓ)}(r) = r^{ℓ+l'} × [polynomial in z of degree n + p]`. The target
function is `Q_{n''}^{(k'',ℓ'')} ∝ r^{ℓ''} P_{n''}^{(α+k'', ℓ''+½)}(z)`. By the parity rule `ℓ + l' − ℓ'' = 2κ`
is an even, non-negative integer, so `r^{ℓ+l'} = r^{ℓ''} (r²)^κ = r^{ℓ''} ((1+z)/2)^κ` and the re-expansion is
that of a **polynomial in z of degree n + p + κ** in the family `(α+k'', ℓ''+½)`. Nothing non-polynomial ever
appears — the ball is no worse than the shell here.

**Band.** Write `Δ = ℓ'' − ℓ ∈ [−l', l']`, `κ = (l' − Δ)/2`, `κ' = (l' + Δ)/2` (non-negative integers,
`κ + κ' = l'`). Upper limit: degree counting gives `n'' ≤ n + p + κ`. Lower limit: the coefficient is
`∫ (1+z)^κ f(z) P_n^{(a,b)} P_{n''}^{(a,b'')} (1−z)^a (1+z)^{b''} dz` with `b = ℓ+½`, `b'' = b + Δ`, i.e.
`∫ [(1+z)^{κ+Δ} f P_{n''}^{(a,b'')}] P_n^{(a,b)} (1−z)^a (1+z)^b dz`, and `κ + Δ = κ'`; orthogonality of
`P_n^{(a,b)}` against lower degrees kills `n'' < n − p − κ'`. Hence

    n'' − n ∈ [ −(p + κ'),  +(p + κ) ],    width  2p + l' + 1  =  d_r + 1,   d_r = 2p + l'  (degree in r),

independent of `ℓ`, `n`, `m`, and of `Δ` except for the asymmetric placement of the band. The `k`-conversion
`E(+1)^{dk}` (upper bidiagonal per unit, `jacobi.py:370-392`) adds `dk` diagonals. In stored index the block
is additionally offset by `⌊ℓ''/2⌋ − ⌊ℓ/2⌋`. For comparison, on the shell the same NCC mode (degree `d_r` in
`r`, Chebyshev/Jacobi in `r`) gives blocks of width `2d_r + 1`.

**Operator realisation with the existing Zernike operators** (all banded; `zernike.py:45-121`,
`jacobi.py:370-400`). With `B(±1)` the Jacobi `b`-parameter raising/lowering operators:
`r · Q_n^{(k,ℓ)} = Σ_{n' ∈ {n−1,n}} [R(+1)]_{n'n} Q_{n'}^{(k,ℓ+1)}` and
`r · Q_n^{(k,ℓ)} = Σ_{n' ∈ {n,n+1}} [R(−1)]_{n'n} Q_{n'}^{(k,ℓ−1)}`, so

    r^{l'} : ℓ → ℓ''   =   R(sgn Δ)^{|Δ|}  ·  (R(−1) R(+1))^{(l' − |Δ|)/2},

which is precisely `BallRadialBasis.radius_multiplication_matrix(ell, regtotal, order=Δ, d=l'−|Δ|)`
(`basis.py:4224-4239`), the routine the radial-only NCC code already uses for regularity mismatches with
`d = regtotal_ncc − |diff_regtotal|` (`4267-4271`). The polynomial factor
`(a_{l'}/r^{l'})(z) = Σ_p c_{l'p} [Q_p^{(k_a,l')}/r^{l'}](z)` is evaluated at the argument's Jacobi matrix
`J_ℓ = operator_matrix('Z', ell, regtotal)` by `clenshaw.matrix_clenshaw` with the three-term recurrence of the
NCC family `(α+k_a, l'+½)` and `f0 = Q_0^{(k_a,l')}(r=1)`, exactly as in `basis.py:4272-4275`. Finally
`conversion_matrix(ℓ'', regtotal_out, dk)` (`4216-4222`). Per pencil `m`:

    B^{(l')}_{ℓ→ℓ''} = E^{dk}|_{ℓ''} · [r^{l'}]_{ℓ→ℓ''} · (a_{l'}/r^{l'})(J_ℓ),
    M_{(ℓ'',n''),(ℓ,n)} = Σ_{l'} G^m_{ℓ l' ℓ''} · B^{(l')}_{ℓ→ℓ''}[n'', n].

The radial blocks `B` do not depend on `m`; only `G` does.

**Counting.** A row `(ℓ'', n'')` receives blocks from `ℓ ∈ [ℓ'' − L', ℓ'' + L']` (`2L' + 1` of them), each with at
most `d_r + 1` entries; different `l'` of the same parity land in the same block and their bands overlap. For
`L' = 4`, degree `≤ 8` (so `p_max = ⌊(8 − l')/2⌋`): `Δ = 0: 9`, `±1: 8`, `±2: 9`, `±3: 8`, `±4: 9`, total
`9 + 16 + 18 + 16 + 18 = 77` — the measured maximum. Shell: `9 × 17 = 153`, also the measured maximum.

**Tensors / spin-weighted NCCs.** On the ball a tensor component is stored in regularity components
`ρ ∈ {−1, +1, 0}^rank` and its radial family is `Q_n^{(k, ℓ + ρ_tot)}` (transform plans use `ℓ + regtotal`,
`basis.py:4172-4198`, `transforms.py:1508-1560`), i.e. it behaves like `r^{ℓ+ρ_tot}`. The construction above
goes through unchanged with `ℓ → ℓ + ρ_arg`, `l' → l' + ρ_ncc`, `ℓ'' → ℓ'' + ρ_out` in the radial part, the
selection test `d = (l' + ρ_ncc) − |(ℓ'' + ρ_out) − (ℓ + ρ_arg)| ≥ 0`, even (blocks failing it are zero, as in
`basis.py:4271-4291`), and the angular factor replaced by the spin-weighted coupling that the S2 tensor machinery
already computes (`Gamma` intertwiners in `Basis._last_axis_field_ncc_matrix`, `basis.py:285-337`; conjugation
by `radial_recombinations`, `basis.py:3706-3718`, as at `4708-4717`). One subtlety that does *not* carry over
from the shell: the shell converts the NCC to spin components before the angular coupling (`4698-4699`) because
its radial basis is regularity-independent. On the ball the NCC's regularity components at fixed `l'` live in
different radial families, so the loop must run over NCC regularity components `(ρ_ncc, l', p)`, applying the
`Q_{l'}` intertwiner to a unit regularity vector to obtain the spin-component weights fed to the `Gamma`
coupling. The bandwidth law is unchanged (`d_r = 2p + l' + ρ_ncc`).

## 4. Bandwidth and cost, target resolution

`N_r = 32`, `l_max = 63`, complex, `m = 0` pencil, NCC with `l' ≤ 4` and polynomial degree `≤ 8` in `r`
(random coefficients; identical truncations on the ball and the shell). Shell and radial-only numbers are
Dedalus's own matrices (`store_ncc_matrices`); the ball meridional matrix is the prototype's.

| operator | rows (valid) | nnz | nnz/row mean | nnz/row max | block width |
|---|---|---|---|---|---|
| ball, radial-only NCC (Dedalus) | 1056 | 8,264 | 7.8 | 9 | `d_r + 1 = 9` |
| **ball, meridional NCC (prototype)** | **1056** | **68,138** | **64.5** | **77** | `d_r + 1` per (ℓ→ℓ'') block, 9 blocks per row |
| shell, meridional NCC (Dedalus) | 2048 | 262,432 | 128.1 | 153 | `2d_r + 1 = 17` per block, 9 blocks per row |

Fill fraction of the ball meridional pencil relative to dense: 6.1 %. The ball pencil has ~half the rows of
the shell's at equal `(N_r, l_max)` because of the triangular truncation `Σ_ℓ (N_max + 1 − ⌊ℓ/2⌋)`. Number of
distinct radial blocks for `l' ≤ 4`: `Σ_{l'} (l'+1)` per ℓ = 15, so 927 for 64 ℓ's (some fall outside the
triangle); they are `m`-independent and reusable across all `m` pencils. The unoptimised Python prototype
assembles them in 23 s (Clenshaw with ≤ 5 terms on padded 129×129 sparse matrices; a cached, banded
implementation would be one to two orders faster). For a problem with many variables the dominant cost is the
sparse LU of the ℓ-coupled pencils, the same cost class as the shell's meridional problems; the ball has the
smaller pencil.

## 5. Prototype and verification

`dedalus_ball_meridional_ncc_prototype.py`, function `build_matrix(m, f, g, b_ncc, b_arg, b_out)`, builds the
coupled-ℓ matrix for one `m` in the subsystem's coefficient layout using only existing Dedalus calls
(`SphereBasis._last_axis_component_ncc_matrix`, `BallRadialBasis.operator_matrix('Z')`,
`radius_multiplication_matrix`, `conversion_matrix`, `clenshaw.jacobi_recursion`, `clenshaw.matrix_clenshaw`,
`zernike.polynomials`). Scalar NCC × scalar field, `complex128`.

Reference: the exact Galerkin projection of the grid-space product. Because `DEALIAS_BEFORE_CONVERTING = True`
(`dedalus/dedalus.cfg:41`) makes a forward transform into a `k > 0` basis truncate *before* converting, the
naive reference `Convert((f*g).evaluate(), k_out)` is not the `k_out` projection (the dropped `n ≥ N` modes feed
the top rows through the bidiagonal `E`); the script therefore transforms the product on a `2N_r` radial basis
sharing the same dealiased quadrature grid and truncates. Results (`max |M g − P(a u)| / max |P(a u)|`):

| case | (Nφ, Nθ, N_r) | R | k_ncc | k_arg | k_out | rel. error |
|---|---|---|---|---|---|---|
| `a = r cos θ` (l'=1, p=0) | (8,8,8) | 1.0 | 0 | 0 | 0 | 1.4e-15 |
| `a = r² P_2(cos θ)` (l'=2, p=0) | (8,8,8) | 1.5 | 0 | 0 | 0 | 1.3e-15 |
| `a = r³(1−r²) cos θ + r³ P_3(cos θ)` | (8,8,8) | 1.0 | 0 | 0 | 0 | 1.6e-15 |
| random full-resolution NCC | (8,8,8) | 1.0 | 0 | 0 | 0 | 5.2e-15 |
| random NCC | (8,8,8) | 1.3 | 0 | 1 | 1 | 8.9e-15 |
| random NCC | (8,8,8) | 1.3 | 2 | 0 | 2 | 2.5e-15 |
| random NCC | (8,12,12) | 2.0 | 1 | 1 | 1 | 1.4e-13 |

Measured block structure (Zernike-index offsets `n'' − n`): for `a = r cos θ`, blocks `(ℓ → ℓ+1)` occupy
`[−1, 0]` and `(ℓ → ℓ−1)` occupy `[0, +1]` (bidiagonal, `d_r + 1 = 2`); for `a = r² P_2`, `(ℓ → ℓ+2)`: `[−2, 0]`,
`(ℓ → ℓ)`: `[−1, +1]`, `(ℓ → ℓ−2)`: `[0, +2]` (`d_r + 1 = 3`) — exactly `[−(p+κ'), p+κ]`.

End-to-end: with the four patches of section 6 installed, `-lap(u) + (1 + ½ r cos θ) u + lift(τ) = F`,
`u(r=1) = 0` at (8,8,8) builds subproblems `(m, None, None)`, the `m = 0` LHS is 60×60 (52 valid `u` modes +
8 `τ` modes; the padded storage has 64 slots), solves in 1.9 s, satisfies the boundary condition to 7e-18, and
its grid-evaluated residual is 1.1e-10, equal to `u`'s last radial coefficient (the resolution floor of the
grid product at `N_r = 8`).

## 6. Implementation sketch (mirroring the shell)

1. **`BallBasis.meridional_basis`** (property): `return self.clone_with(shape=(1,) + self.shape[1:])`
   — identical to `ShellBasis.meridional_basis` (`basis.py:4626-4630`).

2. **`BallBasis.build_ncc_matrix(product, subproblem, ncc_cutoff, max_ncc_terms)`**, mirroring
   `ShellBasis.build_ncc_matrix` (`4667-4681`): constant NCC → `super()`; radial NCC (`shape[:2] == (1,1)`) →
   if `subproblem.group[axis-1] is None` (ℓ-coupled problem) build the block-diagonal over ℓ of the existing
   per-ℓ matrices, padded to the subsystem layout, otherwise `super()`; meridional (`shape[0] == 1`) →
   `build_meridional_ncc_matrix`; else `NotImplementedError("Azimuthal NCCs not yet supported.")`.
   The radial-in-coupled case needs `BallRadialBasis._last_axis_component_ncc_matrix` (`4245-4293`) to take ℓ
   as an argument instead of `subproblem.group[1]` (line 4247).

3. **`BallBasis.build_meridional_ncc_matrix`** (new; the prototype's `build_matrix` is a working scalar
   version):
   * `ells = arange(|m|, Lmax+1)`, reversed when `S2_basis.ell_reversed(dist)[m]` (as at `4709-4712`);
     `Nsub = N_max + 1 − ⌊|m|/2⌋`; block `(ℓ'', ℓ)` occupies rows `i'' Nsub + ⌊ℓ''/2⌋ − ⌊|m|/2⌋ + [0, n_size(ℓ''))`
     and the analogous columns.
   * Loop over NCC regularity components `ρ_ncc`, angular degree `l'`, with radial coefficient vector
     `c = ncc_data[ρ_ncc, 0, l', n_slice(l')]`; skip if `‖c‖ ≤ ncc_cutoff` (this is where a meridional NCC with
     few angular modes stays cheap: the ℓ-bandwidth of the assembled matrix is the *retained* `L'`, not `l_max`).
   * Angular coupling for that `(l', ρ_ncc)`: the S2 tensor NCC matrix for a one-hot coefficient at `l'`
     (its spin components from `Q_{l'}^T`), via `S2_basis._last_axis_field_ncc_matrix` as at `4706`,
     conjugated with `Qout`/`Qarg` per ℓ (`4708-4717`); result `G[(ρ_out, ℓ''), (ρ_arg, ℓ)]`, `m`-dependent, cheap.
   * Radial blocks for each nonzero `G` entry: `Δ = (ℓ''+ρ_out) − (ℓ+ρ_arg)`, `d = (l'+ρ_ncc) − |Δ|`; if `d < 0`
     or odd → zero; else `E^{dk}|_{ℓ''+ρ_out} · radius_multiplication_matrix(ℓ, ρ_arg, Δ, d, size=Nmat) ·
     matrix_clenshaw(c, A, B, f0)` with `(A, B) = jacobi_recursion(len(c), α+k_ncc, l'+ρ_ncc+½, J_{ℓ+ρ_arg})`,
     `f0 = zernike.polynomials(3, 1, α+k_ncc, l'+ρ_ncc, 1)[0]·I`, truncated to `(n_size(ℓ''), n_size(ℓ))`.
     Cache these blocks on the radial basis keyed by `(id of NCC data, ρ_ncc, l', ℓ, ρ_arg, ℓ'', ρ_out, dk)`;
     they are `m`-independent.
   * `Nmat`: the kept `N × N` block is exact iff `Nmat ≥ N + p_max + l' + dk + 1`. The existing heuristic
     `Nmat = 3(N0+1)/2 + …` (`4270`) implicitly assumes an NCC of radial degree `≤ N/2`; for meridional NCCs
     use the explicit bound (cheap, the matrices are tiny).
   * `ncc_cutoff` semantics: per `(l', ρ_ncc)` vector norm to skip whole angular modes, then per-`p` inside
     `matrix_clenshaw` (already implemented), plus the existing `|Gamma| > cutoff` test on tensor components.
     `max_ncc_terms` can cap `p`.
   * Real dtype: same status as the shell (`xfail "Real meridional NCCs not implemented"`,
     `test_spherical_ncc.py:167`); the cos/msin block form `[[C, −S], [S, C]]` of `4279-4283` applies.

4. **`SphericalEllOperator._coupled_ell_matrices`** (`operators.py:3225-3233`): embed each ℓ block into the
   padded radial storage, `P_out(ℓ) @ A_ℓ @ P_in(ℓ)^T` with `P(ℓ) = eye(Nsub, n_size(ℓ), k = −(⌊ℓ/2⌋ − ⌊|m|/2⌋))`
   (identity for 2-D bases and on the shell, where `_nmin = 0`). Alternatively make the subsystem layout compact
   for the ball; but `field_size`, `valid_modes`, `evaluate_as_ncc` and `Subsystem.gather/scatter` all use the
   padded layout today, so padding the operators is the smaller change.

5. **`LiftBall.subproblem_matrix`** (`basis.py:5215-5247`): rewrite in the `LiftShell` form (`5328-5347`):
   `super().subproblem_matrix` (generic ℓ-coupled assembly) followed by the interleaved `Q_ℓ^T` conversion; add the
   `S2_basis` attribute `LiftShell` has (`5321-5323`). `LiftBallRadius` (`5258`) has the same structure.
   `BallRadialInterpolate` (`5896`), `ConvertSpherical3D`, `SphericalLaplacian` etc. already inherit the generic
   path and worked in the demo.

6. **`Spherical3DBasis.valid_elements`** (`basis.py:4440-4463`), coefficient branch: for ball bases mark
   `n < _nmin(ℓ)` invalid (`m, l, n = elements_to_groups(...)`, then `valid[..., n < ⌊l/2⌋] = False`). Without
   this the padded slots are counted as valid modes (64 instead of 52 for `m = 0` at (8,8,8)) and `splu` reports
   "Factor is exactly singular".

7. **Tests**: remove the three `xfail` markers at `test_spherical_ncc.py:160,193,226` (multiply, dot, cross);
   add ball versions of `test_lap_meridional_ncc_shell` and `test_lap_meridional_radial_ncc_shell`
   (`tests/test_lbvp.py:515,545`).

## 7. Subtleties worth stating explicitly

* **Regularity of the NCC.** A meridional NCC on the ball is automatically regular: its `l'` component is
  `r^{l'} × (even polynomial)` by construction of the transform. If NCC data are written in coefficient space
  by hand, the basis enforces the same structure. No extra work.
* **Parity.** The Gaunt rule `ℓ + l' + ℓ''` even is what makes `r^{ℓ+l'−ℓ''}` an even power and keeps everything
  polynomial. For spin-weighted couplings the analogous condition is the `d ≥ 0, even` test on
  `(l' + ρ_ncc) − |Δ + ρ_out − ρ_arg|`; blocks failing it carry zero weight for regular fields (the existing
  radial code already zeroes them, `4286-4291`).
* **Dealiasing.** The matrix route is an exact projection (given `Nmat`) and needs no dealiasing at all; the
  grid route (explicit evaluation of `a·u`) is exact under the usual 3/2 rule in `r` and `θ` when `a` is resolved
  by at most half the modes in each direction — the same requirement as on the shell, nothing ball-specific.
  For verification against grid products with `k_out > 0` remember the `DEALIAS_BEFORE_CONVERTING`
  truncation-order issue described in section 5.
* **Cost.** Assembly is negligible once the `m`-independent radial blocks are cached (`~ (l_max+1) Σ_{l'≤L'}(l'+1)`
  small banded matrices). Solve cost is that of ℓ-coupled pencils, as for shell meridional problems; the ball
  pencil is about half the size at equal `(N_r, l_max)`.
* **Pre-existing quirk.** `BallBasis.__mul__` returns `k = 0` while `BallBasis.__rmatmul__` returns `max(k)`
  (`basis.py:4811-4833`), so on the ball `(f*g).evaluate()` and `(f*g).reinitialize(ncc=True)` land in different
  `k` bases whenever either factor has `k > 0` (on the shell both paths give `k_ncc + k_arg`, `basis.py:3887-3900`).
  `test_meridional_multiply` (`test_spherical_ncc.py:168-190`) compares the two coefficient arrays directly, so a
  ball version of it must convert one result into the other's basis first.
