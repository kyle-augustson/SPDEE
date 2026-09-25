"""
Prototype of meridional (theta, r) NCC multiplication on the Dedalus-3 BallBasis.

Accompanies docs/dedalus_ball_meridional_ncc.md.  Tested against Dedalus 3.0.5 (upstream master,
commit 2a2ad151) with complex128 fields and scalar NCC / scalar argument.

Contents
  build_matrix(...)         coupled-ell NCC matrix for one m, built only from existing Dedalus calls
                            (SphereBasis Clenshaw for the Gaunt coupling, BallRadialBasis zernike
                            operators + tools.clenshaw for the radial blocks)
  install_patches()         the four monkeypatches that make an ell-coupled BallBasis problem run
                            end-to-end (these are the implementation sketch, in executable form)
  verify(...)               compare M @ g against the exact projection of the grid-space product
  lbvp_demo()               Poisson-type LBVP  -lap(u) + a(r,theta) u = F  on the ball, solved by d3.LBVP
  nnz_count()               nnz/row at N_r=32, l_max=63, NCC with l'<=4 and radial degree <= 8, versus
                            Dedalus's own shell-meridional and ball-radial NCC matrices

Run:  OMP_NUM_THREADS=1 python dedalus_ball_meridional_ncc_prototype.py [--count]
"""
import sys, time, types
import numpy as np
from scipy import sparse
import dedalus.public as d3
from dedalus.core import basis as dbasis
from dedalus.core import operators as dops
from dedalus.tools import clenshaw
from dedalus.tools.array import interleave_matrices
from dedalus.libraries import dedalus_sphere


# ----------------------------------------------------------------------------------------------
# 1. The NCC matrix
# ----------------------------------------------------------------------------------------------

def gaunt_matrix_S2(m, lp, b_ncc, b_arg, b_out, cutoff=1e-6):
    """S2 product matrix over ell (fixed m) of the single scalar NCC mode Y_{l'}^0, in Dedalus's own
    SWSH normalisation: obtained by feeding a unit coefficient vector to SphereBasis's Clenshaw."""
    S2n, S2a, S2o = b_ncc.S2_basis(), b_arg.S2_basis(), b_out.S2_basis()
    e = np.zeros(S2n.ell_size(0), dtype=np.complex128); e[lp] = 1.0
    sp = types.SimpleNamespace(group=(m, None, None))
    G = dbasis.SphereBasis._last_axis_component_ncc_matrix(sp, S2n, S2a, S2o, e, (), (), (), (), (), (), cutoff)
    return G.tocsr()


def radial_block(rb_arg, rb_ncc, c, lp, ell, lpp, dk, Nmat, cutoff=1e-6):
    """Radial matrix of multiplication by a_{l'}(r) = sum_p c_p Q_p^{(k_ncc, l')}(r), mapping the argument
    family Q_n^{(k_arg, ell)} to the output family Q_{n''}^{(k_out, l'')}.  Only existing calls are used.

    Structure:  conv(E^dk at l'') @ [ r^{l'} : ell -> l'' ] @ [ (a_{l'}/r^{l'})(J_ell) ]
      * (a_{l'}/r^{l'}) is a polynomial in z = 2r^2-1 of degree p_max; evaluated at the Jacobi matrix
        J_ell of the argument family by Clenshaw -> banded, half-width p_max.
      * r^{l'} = r^{|Delta|} (r^2)^{d/2},  Delta = l'' - ell,  d = l' - |Delta|  (even, >= 0 by the Gaunt
        parity rule): R(+-1)^{|Delta|} then (R(-1) R(+1))^{d/2} -> banded, width l'+1.
      * E(+1)^dk -> banded, width dk+1.
    """
    Delta = lpp - ell
    d = lp - abs(Delta)
    assert d >= 0 and d % 2 == 0, (lp, ell, lpp)
    a_ncc = rb_ncc.alpha + rb_ncc.k
    b_ncc = lp + 1/2
    J = rb_arg.operator_matrix('Z', ell, 0, size=Nmat)
    A, B = clenshaw.jacobi_recursion(len(c), a_ncc, b_ncc, J)
    f0 = dedalus_sphere.zernike.polynomials(3, 1, a_ncc, lp, 1)[0] * sparse.identity(Nmat)
    poly = clenshaw.matrix_clenshaw(c, A, B, f0, cutoff=cutoff)
    rfac = rb_arg.radius_multiplication_matrix(ell, 0, Delta, d, size=Nmat)
    conv = rb_arg.conversion_matrix(lpp, 0, dk, size=Nmat)
    return (conv @ rfac @ poly).tocsr()


def build_matrix(m, f, g, b_ncc, b_arg, b_out, cutoff=1e-6, Nmat=None, lp_max=None):
    """Coupled-ell NCC matrix for subproblem group (m, None, None), in the subsystem's coefficient layout:
    ells in Dedalus's order for this m (reversed for negative m), radial storage of length N0 - nmin(|m|)
    per ell with Zernike index n at position n + nmin(ell) - nmin(|m|).  Slots with n < nmin(ell) are empty."""
    dist = f.dist
    rb_arg, rb_ncc = b_arg.radial_basis, b_ncc.radial_basis
    Lmax, Nmax = b_arg.Lmax, rb_arg.Nmax
    N0 = Nmax + 1
    dk = b_out.k - b_arg.k
    if Nmat is None:
        Nmat = 2*N0 + Lmax + dk + 2      # generous; N0 + p_max + l'_max + dk + 1 suffices
    ells = np.arange(abs(m), Lmax+1)
    if b_arg.S2_basis().ell_reversed(dist)[m]:
        ells = ells[::-1]
    ells = list(ells)
    nell = len(ells)
    nmin_m = rb_arg._nmin(abs(m))
    Nsub = N0 - nmin_m
    fc = f['c']
    ncc_ellmap = {ell: (mi, li) for ell, mi, li in b_ncc.ell_maps(dist)}
    if lp_max is None:
        lp_max = Lmax
    rows, cols, vals = [], [], []
    block_stats = {}
    for lp in range(0, lp_max+1):
        mi, li = ncc_ellmap[lp]
        c = fc[mi, li, rb_ncc.n_slice(lp)].ravel()
        if np.max(np.abs(c)) <= cutoff:
            continue
        G = gaunt_matrix_S2(m, lp, b_ncc, b_arg, b_out, cutoff)
        for i, ell in enumerate(ells):
            for ipp, lpp in enumerate(ells):
                gval = G[ipp, i]
                if abs(gval) <= 1e-14:
                    continue
                if not (abs(ell-lp) <= lpp <= ell+lp and (ell+lp+lpp) % 2 == 0):
                    raise RuntimeError(f"non-Gaunt coupling from S2 Clenshaw: l'={lp}, ell={ell}, l''={lpp}")
                R = radial_block(rb_arg, rb_ncc, c, lp, ell, lpp, dk, Nmat, cutoff)
                nout, nin = rb_arg.n_size(lpp), rb_arg.n_size(ell)
                R = R[:nout, :nin].tocoo()
                keep = np.abs(R.data) > 1e-14
                rows.append(ipp*Nsub + rb_arg._nmin(lpp) - nmin_m + R.row[keep])
                cols.append(i*Nsub + rb_arg._nmin(ell) - nmin_m + R.col[keep])
                vals.append(gval * R.data[keep])
                off = R.row[keep] - R.col[keep]
                block_stats[(lp, int(ell), int(lpp))] = (int(off.min()), int(off.max()), int(keep.sum())) if off.size else (0, 0, 0)
    if rows:
        M = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(nell*Nsub, nell*Nsub)).tocsr()
    else:
        M = sparse.csr_matrix((nell*Nsub, nell*Nsub), dtype=np.complex128)
    M.sum_duplicates()
    return M, ells, block_stats


# ----------------------------------------------------------------------------------------------
# 2. The four changes Dedalus needs (as monkeypatches; see the brief for where they belong)
# ----------------------------------------------------------------------------------------------

def install_patches():
    # (1) BallBasis.meridional_basis, mirroring ShellBasis.meridional_basis (basis.py:4627)
    dbasis.BallBasis.meridional_basis = property(lambda self: self.clone_with(shape=(1,) + self.shape[1:]))

    # (2) BallBasis.build_ncc_matrix, mirroring ShellBasis.build_ncc_matrix (basis.py:4667);
    #     currently BallBasis inherits Basis.build_ncc_matrix which raises at basis.py:276
    def ball_build_ncc_matrix(self, product, subproblem, ncc_cutoff, max_ncc_terms):
        axis = product.dist.last_axis(self)
        ncc_basis = product.ncc.domain.get_basis(axis)
        if ncc_basis is None or ncc_basis.shape[:2] == (1, 1):
            if subproblem.group[axis-1] is None and ncc_basis is not None:
                raise NotImplementedError("radial NCC in an ell-coupled ball subproblem: needs per-ell block-diag "
                                          "(BallRadialBasis._last_axis_component_ncc_matrix assumes group[1] is ell)")
            return dbasis.Basis.build_ncc_matrix(self, product, subproblem, ncc_cutoff, max_ncc_terms)
        if ncc_basis.shape[0] != 1:
            raise NotImplementedError("Azimuthal NCCs not yet supported.")
        arg_basis = product.operand.domain.get_basis(axis)
        out_basis = product.domain.get_basis(axis)
        M, ells, st = build_matrix(subproblem.group[axis-2], product.ncc, product.operand, ncc_basis, arg_basis, out_basis, cutoff=ncc_cutoff)
        return M
    dbasis.BallBasis.build_ncc_matrix = ball_build_ncc_matrix

    # (3) SphericalEllOperator._coupled_ell_matrices (operators.py:3225): embed each ell block into the
    #     subproblem's padded radial storage (N0 - nmin(|m|)); identity on the shell where nmin = 0.
    def _coupled_ell_matrices_padded(self, regindex_in, regindex_out, m):
        S2 = self.S2_basis
        ell_list = np.arange(abs(m), S2.Lmax+1)
        if S2.ell_reversed(self.dist)[m]:
            ell_list = ell_list[::-1]
        def pad(basis_, ell, n):
            if basis_.dim == 2:
                return sparse.eye(1, n)
            rb = basis_.get_radial_basis()
            off = rb._nmin(ell) - rb._nmin(abs(m)); Nsub = rb.Nmax + 1 - rb._nmin(abs(m))
            return sparse.eye(Nsub, n, k=-off)
        mats = []
        for ell in ell_list:
            A = sparse.csr_matrix(self._wrap_radial_matrix(regindex_in, regindex_out, ell, return_zeros=True))
            mats.append(pad(self.output_basis, ell, A.shape[0]) @ A @ pad(self.input_basis, ell, A.shape[1]).T)
        return sparse.block_diag(mats, format='csr')
    dops.SphericalEllOperator._coupled_ell_matrices = _coupled_ell_matrices_padded

    # (4a) LiftBall.subproblem_matrix (basis.py:5215) rewritten in the LiftShell (basis.py:5328) style
    def liftball_subproblem_matrix(self, subproblem):
        matrix = dops.SphericalEllOperator.subproblem_matrix(self, subproblem)
        m = subproblem.group[self.last_axis - 2]; ell = subproblem.group[self.last_axis - 1]
        if ell is None:
            ell_list = np.arange(abs(m), self.input_basis.Lmax + 1)
            if self.input_basis.ell_reversed(self.dist)[m]:
                ell_list = ell_list[::-1]
        else:
            ell_list = [ell]
        Q_dict = self.output_basis.radial_basis.radial_recombinations(self.tensorsig, ell_list=tuple(ell_list))
        Q_forward = interleave_matrices([Q_dict[ell].T for ell in ell_list])
        return sparse.csr_matrix(matrix @ Q_forward)
    dbasis.LiftBall.subproblem_matrix = liftball_subproblem_matrix
    dbasis.LiftBall.S2_basis = property(lambda self: self.output_basis.S2_basis())

    # (4b) Spherical3DBasis.valid_elements (basis.py:4432): in coeff space, ball slots with n < nmin(ell) must be
    #      invalid (ell-separable subproblems never see them because their radial slice is n_slice(ell)).
    _orig_valid_elements = dbasis.Spherical3DBasis.valid_elements
    def valid_elements_ball(self, tensorsig, grid_space, elements):
        valid = _orig_valid_elements(self, tensorsig, grid_space, elements)
        if not grid_space[2] and isinstance(self, dbasis.BallBasis):
            m, l, n = self.elements_to_groups(grid_space, elements)
            bad = np.ma.getdata(n) < self.radial_basis._nmin(np.ma.getdata(l))
            valid[..., bad] = False
        return valid
    dbasis.Spherical3DBasis.valid_elements = valid_elements_ball


# ----------------------------------------------------------------------------------------------
# 3. Verification against the grid-space product
# ----------------------------------------------------------------------------------------------

def verify(Nphi, Ntheta, Nr, radius, k_ncc, k_arg, ncc_func=None, dealias=3, seed=0, label=""):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=np.complex128)
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius, dealias=dealias, dtype=np.complex128)
    b_ncc = b.clone_with(shape=(1,)+b.shape[1:], k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(name='f', bases=b_ncc); g = d.Field(name='g', bases=b_arg)
    phi, theta, r = d.local_grids(b)
    if ncc_func is None:
        f.fill_random('g', seed=seed)
    else:
        f['g'] = ncc_func(theta, r)
    g.fill_random('g', seed=seed+1)
    problem = d3.LBVP([g]); problem.add_equation((f*g, 0))
    solver = problem.build_solver()
    w1 = (f*g).reinitialize(ncc=True, ncc_vars=[g])
    b_out = w1.domain.get_basis(c)
    # Exact Galerkin projection of the grid product onto the k_out basis: transform on a 2x larger radial basis
    # sharing the same dealiased quadrature grid (avoids the k=0 truncation before k-conversion), then truncate.
    b_big = b_out.clone_with(shape=(Nphi, Ntheta, 2*Nr))
    h = d.Field(bases=b_big); h.change_scales((dealias, dealias, dealias/2))
    f.change_scales(dealias); g.change_scales(dealias)
    h['g'] = f['g'] * g['g']
    ref = d.Field(bases=b_out); ref['c'] = 0
    ref['c'][:] = h['c'][:, :, :Nr]
    f.change_scales(1); g.change_scales(1)
    errs = []; stats_all = {}
    for sp in solver.subproblems:
        m = sp.group[0]
        M, ells, stats = build_matrix(m, f, g, b_ncc, b_arg, b_out)
        stats_all[m] = stats
        ss = sp.subsystems[0]
        gv = g['c'][ss.field_slices(g)].ravel()
        rv = ref['c'][ss.field_slices(ref)].ravel()
        errs.append(np.max(np.abs(M @ gv - rv)))
    refmax = np.max(np.abs(ref['c']))
    print(f"  {label:38s} (Nphi,Ntheta,Nr)=({Nphi},{Ntheta},{Nr}) R={radius} k_ncc={k_ncc} k_arg={k_arg} k_out={b_out.k}: "
          f"max|M g - P(a u)| = {max(errs):.2e}, max|P(a u)| = {refmax:.2e}, rel {max(errs)/refmax:.1e}")
    return max(errs), refmax, stats_all


def run_verification():
    print("== Verification against Dedalus's grid-space product (complex128, scalar NCC x scalar field) ==")
    e, rm, st = verify(8, 8, 8, 1.0, 0, 0, ncc_func=lambda th, r: r*np.cos(th), label="a = r cos(theta)  [l'=1, p=0]")
    print("     blocks (l', ell, l''): band offsets n''-n [min,max], nnz  (first four):")
    for key, v in sorted(st[0].items())[:4]:
        print(f"       {key}: [{v[0]:+d},{v[1]:+d}]  nnz={v[2]}")
    e, rm, st = verify(8, 8, 8, 1.5, 0, 0, ncc_func=lambda th, r: r**2*(3*np.cos(th)**2-1)/2, label="a = r^2 P_2(cos theta)  [l'=2, p=0]")
    for key, v in sorted(st[0].items())[:6]:
        print(f"       {key}: [{v[0]:+d},{v[1]:+d}]  nnz={v[2]}")
    verify(8, 8, 8, 1.0, 0, 0, ncc_func=lambda th, r: r**3*np.cos(th)*(1-r**2) + r**3*(5*np.cos(th)**3-3*np.cos(th))/2,
           label="a = r^3(1-r^2)cos + r^3 P_3  [l'=1,3]")
    verify(8, 8, 8, 1.0, 0, 0, label="random full-resolution NCC")
    verify(8, 8, 8, 1.3, 0, 1, label="random NCC, k_arg=1 (dk=0)")
    verify(8, 8, 8, 1.3, 2, 0, label="random NCC, k_ncc=2 (dk=2)")
    verify(8, 12, 12, 2.0, 1, 1, label="random NCC, k_ncc=k_arg=1, R=2")


# ----------------------------------------------------------------------------------------------
# 4. End-to-end LBVP through d3.LBVP with the patches installed
# ----------------------------------------------------------------------------------------------

def lbvp_demo(Nphi=8, Ntheta=8, Nr=8):
    print("== LBVP demo: -lap(u) + (1 + 0.5 r cos(theta)) u + lift(tau) = F,  u(r=1) = 0  on the ball ==")
    c = d3.SphericalCoordinates('phi', 'theta', 'r'); d = d3.Distributor(c, dtype=np.complex128)
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), radius=1, dealias=3/2, dtype=np.complex128)
    phi, th, r = d.local_grids(b)
    u = d.Field(name='u', bases=b); tau = d.Field(name='tau', bases=b.S2_basis())
    ncc = d.Field(name='ncc', bases=b.meridional_basis); ncc['g'] = 1 + 0.5*r*np.cos(th)
    F = d.Field(name='F', bases=b); F.fill_random('g', seed=5); F.low_pass_filter(scales=0.5)
    lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.LBVP([u, tau], namespace=locals())
    problem.add_equation("-lap(u) + ncc*u + lift(tau) = F")
    problem.add_equation("u(r=1) = 0")
    t0 = time.time(); solver = problem.build_solver(); solver.solve(); t1 = time.time()
    sp0 = solver.subproblems[0]
    print(f"  subproblem groups: {[sp.group for sp in solver.subproblems][:3]} ...;  m=0 LHS shape {sp0.L_min.shape}, nnz {sp0.L_min.nnz};  build+solve {t1-t0:.1f} s")
    print(f"  m=0 valid modes of u: {int(sp0.valid_modes(u, u.valid_modes).sum())} (= sum_ell n_size(ell) = {sum(b.n_size(l) for l in range(Ntheta))}; padded storage {Ntheta*Nr})")
    res = (-d3.lap(u) + ncc*u - F).evaluate(); res.change_scales(1)
    print(f"  residual of -lap(u)+ncc*u-F evaluated on the grid, interior modes: {np.max(np.abs(res['c'][..., :-1])):.2e}  (max|F| = {np.max(np.abs(F['c'])):.2e})")
    print(f"  size of u's last radial coefficient (resolution floor): {np.max(np.abs(u['c'][..., -1])):.2e};  max|u(r=1)| = {np.max(np.abs(u(r=1).evaluate()['g'])):.1e}")
    print("  (the grid residual is limited by the truncation of a*u at N_r=8: it equals u's last coefficient)")


# ----------------------------------------------------------------------------------------------
# 5. nnz per row at the target resolution
# ----------------------------------------------------------------------------------------------

def _rowstats(M, label, valid_rows=None):
    M = M.tocsr(); M.eliminate_zeros()
    mx = np.max(np.abs(M.data)) if M.nnz else 1
    M.data[np.abs(M.data) < 1e-13*mx] = 0; M.eliminate_zeros()
    per_row = np.diff(M.indptr)
    if valid_rows is not None:
        per_row = per_row[valid_rows]
    per_row = per_row[per_row > 0]
    print(f"  {label}: shape {M.shape}, nnz {M.nnz}, rows {per_row.size}, nnz/row mean {per_row.mean():.1f}, max {per_row.max()}")
    return per_row


def nnz_count(Nphi=4, Ntheta=64, Nr=32, LP=4, DEG=8, cut=1e-6):
    print(f"== nnz per row, m=0 pencil, N_r={Nr}, l_max={Ntheta-1}, NCC with l'<={LP}, polynomial degree in r <= {DEG} ==")
    c = d3.SphericalCoordinates('phi', 'theta', 'r'); d = d3.Distributor(c, dtype=np.complex128)
    rng = np.random.default_rng(0)
    # Ball meridional (prototype)
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), radius=1, dealias=1, dtype=np.complex128)
    b_ncc = b.clone_with(shape=(1,)+b.shape[1:]); rb = b.radial_basis
    f = d.Field(bases=b_ncc); g = d.Field(bases=b); f['c'] = 0
    ellmap = {ell: (mi, li) for ell, mi, li in b_ncc.ell_maps(d)}
    for lp in range(LP+1):
        mi, li = ellmap[lp]; pmax = (DEG - lp)//2
        f['c'][mi, li, rb._nmin(lp):rb._nmin(lp)+pmax+1] = rng.standard_normal(pmax+1)
    t0 = time.time(); M, ells, stats = build_matrix(0, f, g, b_ncc, b, b, cutoff=cut, lp_max=LP); t1 = time.time()
    print(f"  ball meridional prototype: {len(stats)} radial blocks (m-independent, reusable for all m), assembled in {t1-t0:.1f} s (unoptimised Python)")
    Nsub = M.shape[0]//len(ells)
    valid = np.concatenate([np.arange(i*Nsub + rb._nmin(ell), (i+1)*Nsub) for i, ell in enumerate(ells)])
    pr_bm = _rowstats(M, "BALL meridional NCC (prototype)", valid)
    # Shell meridional (Dedalus)
    bs = d3.ShellBasis(c, (Nphi, Ntheta, Nr), radii=(0.6, 1.7), dealias=1, dtype=np.complex128)
    fs = d.Field(bases=bs.meridional_basis); gs = d.Field(bases=bs); fs['c'] = 0
    smap = {ell: (mi, li) for ell, mi, li in bs.meridional_basis.ell_maps(d)}
    for lp in range(LP+1):
        mi, li = smap[lp]; fs['c'][mi, li, :DEG+1] = rng.standard_normal(DEG+1)
    gs.fill_random('g')
    problem = d3.LBVP([gs]); problem.add_equation((fs*gs, 0)); solver = problem.build_solver()
    w1 = (fs*gs).reinitialize(ncc=True, ncc_vars=[gs]); w1.store_ncc_matrices([gs], solver.subproblems, ncc_cutoff=cut)
    sp0 = [sp for sp in solver.subproblems if sp.group[0] == 0][0]
    _rowstats(w1._ncc_matrices[sp0], "SHELL meridional NCC (Dedalus)  ")
    # Ball radial (Dedalus)
    fr = d.Field(bases=b.radial_basis); gr = d.Field(bases=b); fr['c'] = 0
    fr['c'][0, 0, :DEG//2+1] = rng.standard_normal(DEG//2+1); gr.fill_random('g')
    problem = d3.LBVP([gr]); problem.add_equation((fr*gr, 0)); solver = problem.build_solver()
    w2 = (fr*gr).reinitialize(ncc=True, ncc_vars=[gr]); w2.store_ncc_matrices([gr], solver.subproblems, ncc_cutoff=cut)
    mats = [w2._ncc_matrices[sp] for sp in solver.subproblems if sp.group[0] == 0]
    _rowstats(sparse.block_diag(mats), "BALL radial NCC (Dedalus)        ")
    print(f"  fill fraction of the ball meridional pencil relative to dense: {pr_bm.sum()/pr_bm.size**2:.4f}")


if __name__ == '__main__':
    install_patches()
    run_verification()
    lbvp_demo()
    if '--count' in sys.argv:
        nnz_count()
