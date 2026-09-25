"""Entrywise certification of SPDEE's staggered (Arakawa-C) hydro operator
against the seed bcg22 operator of the Bekki cross-validation project.

    python scripts/certify_staggered_vs_seed.py --nr 12 --nt 48 [--no-eig]

The seed is pinned at commit 6ef66bb. Parameters: m = 8, nu = kappa =
2e12 cm^2/s, delta = 1e-6, uniform rotation Omega0 = 2 pi 456 nHz.

Conventions:

  seed   omega V = M V, M in rad/s on the cgs state, q ~ exp(i(m phi - omega t)).
         State [v_r interior r-faces | v_th interior th-faces | v_ph | rho1 |
         s1], every block row-major in (i_r, j_theta) through
         kron(r-op, theta-op).
  spdee  K x = lambda x with B = I, omega = -i lambda so x ~ exp(-lambda t)
         (``EvpResult.eigenvalues_to_omega``); nondimensional with lengths
         R_sun, time 1/Omega0, U = R_sun Omega0, rho_bc, c_p = 1 (solar_eigenmodes.py).
         State [v_r r-edges | v_t th-edges incl. both poles | v_p | rho1 |
         s1] (staggered_general._layout_for); the wall v_r DOFs are
         eliminated symmetrically, P^T A P (_reduce_velocity_dofs); no r^n
         row premultiply on a shell (_ball_premultiply).

  Both codes use exp(+i m phi), 2-point staggered stencils, cell-centred
  theta_j = (j + 1/2) pi/Nt and Nr uniform radial cells on [0.71, 0.985].

  Strict conversion: K_spdee = (i/Omega0) D (P M_seed P^T) D^-1 with the unit
  matrix D = diag(1/U, 1/U, 1/U, 1/rho_bc, 1/c_p), i.e. block (i, j) carries
  c_ij = (i/Omega0) d_i/d_j. The script also fits one complex coefficient per
  seed term (alpha_t, +1 = strict conversion) so that a sign or scale error in
  one physical term shows up as alpha_t != 1, and reports the residual after
  the fit (the discretization difference proper).

Layout difference: SPDEE keeps the pole th-edge v_theta DOFs (2 Nr of them),
the seed does not. SPDEE pins them explicitly (a Dirichlet row, no mass:
``constraints['pinned']``), which is the seed's convention v_theta = 0 at the
poles; the comparison removes them, and the "schur" variant, which folds the
south-pole DOFs into the interior when their rows are nonzero, reduces to the
removal.
"""

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.optimize import linear_sum_assignment

SEED = os.path.expanduser(os.environ.get("SPDEE_SEED_TREE", ""))
for _p in (SEED, os.path.join(SEED, "bcg22")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bcg22  # noqa: E402

from spdee.backgrounds.polytrope import PolytropeBackground  # noqa: E402
from spdee.applications.solar_eigenmodes import make_polytrope_system  # noqa: E402
from spdee.backends.fd2.grid import make_shell_grid  # noqa: E402
from spdee.backends.fd2.staggered_compile import compile_fd2_staggered  # noqa: E402

R_LO, R_HI, M = 0.71, 0.985, 8
RSUN, OMEGA0 = 7.0e10, 2 * np.pi * 456e-9
U = RSUN * OMEGA0
NU, DELTA = 2e12, 1e-6
EK = NU / (RSUN ** 2 * OMEGA0)
VARS = ("vr", "vt", "vp", "rho", "s")
WINDOW = dict(re=2.2, im_lo=-0.6, im_hi=0.1)
TERMS = ("pressure", "buoyancy", "coriolis", "mass_flux", "superadiabatic",
         "viscous", "thermal_diff")
TERMS_INVISCID = TERMS[:5]
TERMS_ZERO = ("doppler", "rot_shear", "baroclinic")   # vanish under uniform rotation
PHYSICS = {"pressure": "pressure gradient", "buoyancy": "buoyancy",
           "coriolis": "Coriolis", "mass_flux": "continuity",
           "superadiabatic": "entropy source", "viscous": "viscous stress",
           "thermal_diff": "thermal diffusion"}


# ---------------------------------------------------------------------------
# operators

def seed_model(Nr, Nt, nu=NU, kappa=NU, delta=DELTA, only=None):
    """bcg22 Model; ``only`` keeps a subset of the term flags. The zero-valued
    doppler diagonals stay on so sp.bmat sees every block column."""
    terms = None
    if only is not None:
        terms = {k: (k in only or k == "doppler") for k in bcg22.DEFAULT_TERMS}
    par = bcg22.Params(m=M, Nr=Nr, Nt=Nt, nu=nu, kappa=kappa, delta=delta,
                       terms=terms)
    return bcg22.Model(par)


def spdee_operator(Nr, Nt, Ek=EK, delta=DELTA):
    """SPDEE staggered pencil K (B = I) on Nr x Nt cells; returns (K, layout)."""
    poly = PolytropeBackground(r_inner=R_LO, r_outer=R_HI, delta=delta)
    system = make_polytrope_system(poly, m_val=M, Ek=Ek, Le=0.0,
                                       entropy_bc="neumann", geometry="shell",
                                       with_grad_lnrho_strain=(Ek != 0.0))
    grid = make_shell_grid(N_r=Nr + 1, N_theta=Nt, r_inner=R_LO, r_outer=R_HI)
    cp = compile_fd2_staggered(system, grid, m=M, order_r=2, order_theta=2,
                               coriolis_pairing="seed")
    K, B = cp.assemble_evp_matrices()
    if Ek != 0.0:
        pinned = np.asarray(cp.constraints["pinned"])
        keep = np.setdiff1d(np.arange(B.shape[0]), pinned)
        eye = sps.eye(B.shape[0], format="csr")
        assert B[pinned].nnz == 0, "the pinned pole rings carry mass"
        assert abs(B[keep] - eye[keep]).max() == 0.0, \
            "B is not the identity off the pinned pole rings"
    return K.tocsr(), cp.layout


# ---------------------------------------------------------------------------
# index maps: seed position k -> (var, i_r, j_theta) and -> SPDEE flat index

class SeedIndex:
    """Seed state bookkeeping and the map into SPDEE's flat state."""

    def __init__(self, Nr, Nt, offsets):
        self.Nr, self.Nt = Nr, Nt
        self.N = int(offsets[-1])
        var = np.empty(self.N, dtype=object)
        ir = np.zeros(self.N, dtype=int)
        jt = np.zeros(self.N, dtype=int)
        # v_r: faces i = 1..Nr-1; v_th: th-faces j = 1..Nt-1; centres i, j
        shapes = {"vr": (Nr - 1, Nt), "vt": (Nr, Nt - 1), "vp": (Nr, Nt),
                  "rho": (Nr, Nt), "s": (Nr, Nt)}
        for b, v in enumerate(VARS):
            n_i, n_j = shapes[v]
            sl = slice(offsets[b], offsets[b + 1])
            assert sl.stop - sl.start == n_i * n_j
            ii, jj = np.meshgrid(np.arange(n_i), np.arange(n_j), indexing="ij")
            var[sl] = v
            ir[sl] = ii.ravel() + (1 if v == "vr" else 0)
            jt[sl] = jj.ravel() + (1 if v == "vt" else 0)
        self.var, self.ir, self.jt = var, ir, jt
        self.block = np.array([VARS.index(v) for v in var])

    def spdee_index(self, layout, vr_has_walls):
        """SPDEE flat index of every seed position. ``vr_has_walls`` is True
        when the SPDEE v_r block still carries the two wall layers (inviscid
        row-overwrite path)."""
        Nt = self.Nt
        off = {f.name: f.offset for f in layout.fields}
        idx = np.zeros(self.N, dtype=int)
        for k in range(self.N):
            v, i, j = self.var[k], self.ir[k], self.jt[k]
            if v == "vr":
                idx[k] = off["v_r"] + (i if vr_has_walls else i - 1) * Nt + j
            elif v == "vt":
                idx[k] = off["v_t"] + i * (Nt + 1) + j
            else:
                idx[k] = off[{"vp": "v_p", "rho": "rho1", "s": "s1"}[v]] + i * Nt + j
        assert len(set(idx.tolist())) == self.N
        return idx

    def row_class(self):
        """'wall' (first/last radial index), 'pole' (first/last theta index),
        else 'interior'; wall wins when both apply."""
        cls = np.full(self.N, "interior", dtype=object)
        for k in range(self.N):
            v, i, j = self.var[k], self.ir[k], self.jt[k]
            wall = i in ((1, self.Nr - 1) if v == "vr" else (0, self.Nr - 1))
            pole = j in ((1, self.Nt - 1) if v == "vt" else (0, self.Nt - 1))
            cls[k] = "wall" if wall else ("pole" if pole else "interior")
        return cls

    def label(self, k):
        return f"{self.var[k]}(i={self.ir[k]}, j={self.jt[k]})"


def unit_scaling(model, si):
    """d_k of x_spdee = D x_seed: velocities 1/U, rho1 1/rho_bc, s1 1/c_p."""
    bg = model.par.background
    d = {"vr": 1 / U, "vt": 1 / U, "vp": 1 / U, "rho": 1 / bg.rho_bc, "s": 1 / bg.cp}
    return np.array([d[v] for v in si.var])


def convert(Mseed, d):
    """Seed M (rad/s) into SPDEE's pencil: (i/Omega0) D M D^-1."""
    return (1j / OMEGA0) * (sps.diags(d) @ Mseed @ sps.diags(1 / d)).tocsr()


def pole_dofs(layout, Nr, Nt):
    """SPDEE flat indices of the north (theta = 0) and south (theta = pi)
    pole v_theta DOFs."""
    off = layout.field("v_t").offset
    north = [off + i * (Nt + 1) for i in range(Nr)]
    south = [off + i * (Nt + 1) + Nt for i in range(Nr)]
    return np.array(north), np.array(south)


def reduce_spdee(K, idx, south, schur):
    """K restricted to the seed positions; with ``schur`` the south-pole DOFs
    are folded in exactly (A_ii - A_is A_ss^-1 A_si)."""
    A = K[idx][:, idx].tocsr()
    if not schur or abs(K[south]).max() == 0.0:
        return A
    Ass = K[south][:, south].toarray()
    Asi = K[south][:, idx].toarray()
    Ais = K[idx][:, south].toarray()
    return (A - sps.csr_matrix(Ais @ np.linalg.solve(Ass, Asi))).tocsr()


# ---------------------------------------------------------------------------
# entrywise comparison

def block_max(A, si, bi, bj):
    ri, cj = np.nonzero(si.block == bi)[0], np.nonzero(si.block == bj)[0]
    sub = A[ri][:, cj]
    return abs(sub).max() if sub.nnz else 0.0


def block_table(A_sp, A_ref, si, rows_cls, title, ref_max=None):
    """Per-block max |A_sp - A_ref| / max |A_sp block| split by row class;
    ``ref_max`` (block -> value) normalises by that block's max instead."""
    D = abs((A_sp - A_ref).tocsr())
    Sm = abs(A_sp).max()
    print(f"\n{title}")
    print(f"  global max|A_sp| = {Sm:.4e}   max|diff|/max|A_sp| = {D.max() / Sm:.3e}")
    print("  block     max|A_sp|    rel(all)  rel(interior)  rel(wall)  rel(pole)")
    out = {}
    for bi, vi in enumerate(VARS):
        ri = np.nonzero(si.block == bi)[0]
        for bj, vj in enumerate(VARS):
            cj = np.nonzero(si.block == bj)[0]
            smax, tmax = block_max(A_sp, si, bi, bj), block_max(A_ref, si, bi, bj)
            if smax == 0.0 and tmax == 0.0:
                continue
            ref = ref_max[(vi, vj)] if ref_max else (smax if smax > 0 else tmax)
            db = D[ri][:, cj]
            rel = {}
            for c in ("all", "interior", "wall", "pole"):
                sub = db if c == "all" else db[rows_cls[ri] == c]
                rel[c] = (sub.max() / ref) if sub.nnz else 0.0
            out[(vi, vj)] = rel["all"]
            print(f"  {vi:>3s},{vj:<4s} {smax:10.3e}  {rel['all']:10.2e}  {rel['interior']:10.2e}"
                  f"     {rel['wall']:9.2e}  {rel['pole']:9.2e}")
    return out


def fit_terms(A_sp, terms, si):
    """Least-squares complex coefficient per seed term, fitted block by block
    (each block's terms jointly) so a term is not dominated by its largest
    block. ``A_sp`` must contain only the physics of ``terms``. Returns
    {term: {block: alpha}}."""
    alpha = {t: {} for t in terms}
    for bi, vi in enumerate(VARS):
        ri = np.nonzero(si.block == bi)[0]
        for bj, vj in enumerate(VARS):
            cj = np.nonzero(si.block == bj)[0]
            present = [t for t in terms if block_max(terms[t], si, bi, bj) > 0]
            if not present:
                continue
            y = A_sp[ri][:, cj].toarray().ravel()
            X = np.stack([terms[t][ri][:, cj].toarray().ravel() for t in present], axis=1)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            for t, a in zip(present, coef):
                alpha[t][(vi, vj)] = a
    return alpha


# ---------------------------------------------------------------------------
# eigenvalues

def in_window(om):
    return (np.abs(om.real) <= WINDOW["re"]) & (om.imag >= WINDOW["im_lo"]) \
        & (om.imag <= WINDOW["im_hi"])


def theta_profile(R, si, bi, bj, ref, js):
    """max |R| over the rows of block (bi, bj) with theta index j, / ref."""
    out = []
    for j in js:
        rows = np.nonzero((si.block == bi) & (si.jt == j))[0]
        cols = np.nonzero(si.block == bj)[0]
        sub = R[rows][:, cols]
        out.append((sub.max() / ref) if sub.nnz else 0.0)
    return out


def dense_omega(A, seed=False):
    """All eigenvalues as omega (Omega0 units) and the solve time, through the
    solver layer's dense census (GPU when available).  The shift sits at
    omega = 0.7 - 0.3i in both conventions."""
    from spdee import solvers
    t0 = time.perf_counter()
    sigma = OMEGA0 * (0.7 - 0.3j) if seed else (0.3 + 0.7j)
    lam = solvers.dense(A, sigma=sigma, backend="auto").lam
    om = lam / OMEGA0 if seed else -1j * lam
    return om, time.perf_counter() - t0


def pair(om_a, om_b):
    """Optimal one-to-one pairing (Hungarian) of two eigenvalue sets; returns
    the pair distances and the unassigned members of the larger set."""
    dist = np.abs(om_a[:, None] - om_b[None, :])
    ia, ib = linear_sum_assignment(dist)
    extra = np.setdiff1d(np.arange(len(om_a)), ia) if len(om_a) > len(om_b) else []
    return dist[ia, ib], om_a[extra]


def rossby_of(om):
    target = -2.0 / (M + 1)
    return om[np.argmin(np.abs(om - target))]


def rossby_sparse(A, guess):
    """Rossby mode of a pencil in SPDEE's convention by shift-invert."""
    lam = spla.eigs(A.tocsc(), k=8, sigma=1j * guess, which="LM",
                    return_eigenvectors=False)
    om = -1j * lam
    return om[np.argmin(np.abs(om - guess))]


def report_spectrum(name, om_all, ref_all, t):
    om, om_ref = om_all[in_window(om_all)], ref_all[in_window(ref_all)]
    dd, extra = pair(om, om_ref)
    grow = om_all[om_all.imag > 1e-6]
    print(f"  {name:36s}: {len(om)} in window ({t:.0f} s); Rossby {rossby_of(om):.5f}; "
          f"paired {len(dd)}: max|d omega| = {dd.max():.3e}, median = {np.median(dd):.3e}; "
          "n(|d omega| < 1e-4, 1e-3, 1e-2) = "
          + ", ".join(str(int((dd < x).sum())) for x in (1e-4, 1e-3, 1e-2))
          + f"; growing (Im > 1e-6): {len(grow)}"
          + (f", max Im = {grow.imag.max():.3e}" if len(grow) else ""))
    if len(extra):
        print("    unpaired in-window eigenvalues: " + ", ".join(f"{z:.4f}" for z in extra[:8]))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=12)
    ap.add_argument("--nt", type=int, default=48)
    ap.add_argument("--no-eig", action="store_true", help="skip the dense eigensolves")
    ap.add_argument("--dense-hybrids", action="store_true",
                    help="also solve the sign-aligned seed and the hybrid operators densely")
    args = ap.parse_args()
    Nr, Nt = args.nr, args.nt
    print(f"certify_staggered_vs_seed: Nr x Nt = {Nr} x {Nt}, m = {M}, nu = kappa = {NU:.1e}, "
          f"delta = {DELTA:.0e}, Ek = {EK:.4e}")

    # ---- (a) operators, sizes, layouts ----
    model = seed_model(Nr, Nt)
    Ms = model.M.tocsr()
    si = SeedIndex(Nr, Nt, model.offsets)
    K, layout = spdee_operator(Nr, Nt)
    K0, layout0 = spdee_operator(Nr, Nt, Ek=0.0)
    sizes_sp = {f.name: f.size for f in layout.fields}
    print("\n(a) sizes")
    print(f"  seed  M : {Ms.shape[0]} x {Ms.shape[0]}, nnz = {Ms.nnz}, blocks {model.sizes}")
    print(f"  spdee K : {K.shape[0]} x {K.shape[0]}, nnz = {K.nnz}, blocks {sizes_sp}")
    print(f"  spdee has (Nr-1) Nt = {sizes_sp['v_r']} v_r unknowns (walls eliminated as the seed) "
          f"but Nr (Nt+1) = {sizes_sp['v_t']} v_t unknowns vs the seed's Nr (Nt-1) = "
          f"{model.sizes[1]}: 2 Nr = {2 * Nr} pole v_theta DOFs are extra")
    idx = si.spdee_index(layout, vr_has_walls=False)
    idx0 = si.spdee_index(layout0, vr_has_walls=True)
    north, south = pole_dofs(layout, Nr, Nt)
    extra = np.setdiff1d(np.arange(K.shape[0]), idx)
    assert np.array_equal(np.sort(extra), np.sort(np.concatenate([north, south])))
    print(f"  pole v_t rows (Dirichlet pin, no mass): max|K| north = {abs(K[north]).max():.3e}, "
          f"south = {abs(K[south]).max():.3e}; "
          f"pole v_t columns into seed positions: north = {abs(K[idx][:, north]).max():.3e}, "
          f"south = {abs(K[idx][:, south]).max():.3e}")
    bg = model.par.background
    print(f"  strict conversion c_ij = (i/Omega0) d_i/d_j, d = 1/U (v), 1/rho_bc (rho1), 1/c_p (s1); "
          f"U = {U:.4e} cm/s, rho_bc = {bg.rho_bc}, c_p = {bg.cp:.4e} erg/g/K")

    d = unit_scaling(model, si)
    A_seed = convert(Ms, d)
    A_drop = reduce_spdee(K, idx, south, schur=False)
    A_schur = reduce_spdee(K, idx, south, schur=True)
    S_pole = (A_schur - A_drop).tocsr()
    A0 = K0[idx0][:, idx0].tocsr()          # inviscid SPDEE, walls and poles dropped
    rows_cls = si.row_class()
    terms = {t: convert(seed_model(Nr, Nt, only=(t,)).M.tocsr(), d) for t in TERMS + TERMS_ZERO}
    total = sum(terms.values())
    print(f"  seed single-term operators sum to the full one: max|diff| = {abs(total - A_seed).max():.2e}; "
          "terms that vanish under uniform rotation: "
          + ", ".join(f"{t} {abs(terms[t]).max():.1e}" for t in TERMS_ZERO))

    # ---- (c) entrywise comparison ----
    print("\n(c) entrywise |A_spdee - c P A_seed P^T| with the strict c (pole v_theta DOFs removed from "
          "SPDEE); rel = max|diff| over the row class / max|A_spdee| of the block")
    block_table(A_drop, A_seed, si, rows_cls, "strict conversion, full operators")

    vmask = si.block <= 2
    smask = si.block == 4
    visc_sp = (A_drop - A0).tocsr()
    alpha = fit_terms(A0, {t: terms[t] for t in TERMS_INVISCID}, si)
    alpha.update(fit_terms(visc_sp, {t: terms[t] for t in TERMS[5:]}, si))
    print("\n  per-term coefficient alpha (least squares per block, inviscid terms on K(Ek=0), "
          "diffusive terms on K(Ek) - K(0); +1 = strict conversion):")
    print("  term            physics             block    alpha")
    sign = {}
    for t in TERMS:
        big = max(alpha[t], key=lambda b: block_max(terms[t], si, VARS.index(b[0]), VARS.index(b[1])))
        sign[t] = float(np.sign(alpha[t][big].real))
        for b, a in alpha[t].items():
            flag = "" if abs(a - sign[t]) < 0.05 else "   (differs from the sign by more than 5%)"
            print(f"  {t:15s} {PHYSICS[t]:19s} {b[0]:>3s},{b[1]:<4s} {a.real:+.6f}{a.imag:+.1e}j{flag}")
        assert abs(alpha[t][big] - sign[t]) < 0.05, (t, alpha[t][big])
    A_al = sum(sign[t] * terms[t] for t in TERMS)
    print("  aligned seed operator: A_al = sum_t sign(alpha_t) c P M_t P^T, signs "
          + ", ".join(f"{t} {sign[t]:+.0f}" for t in TERMS))
    rel_al = block_table(A_drop, A_al, si, rows_cls,
                         "residual after the per-term sign alignment (discretization differences only)")
    R = abs((A_drop - A_al).tocsr())
    js = [0, 1, 2, 3, Nt // 2 - 1, Nt - 2, Nt - 1]
    print("  residual by theta row index of the four worst blocks (max|R| / block max|A_sp|), "
          f"j = {js}:")
    for (vi, vj), _ in sorted(rel_al.items(), key=lambda kv: -kv[1])[:4]:
        bi, bj = VARS.index(vi), VARS.index(vj)
        prof = theta_profile(R, si, bi, bj, block_max(A_drop, si, bi, bj), js)
        print(f"  {vi:>3s},{vj:<4s} " + "  ".join(f"{x:8.2e}" for x in prof))

    # ---- (d) attribution ----
    print("\n(d) attribution")
    inside = np.outer(vmask, vmask) | np.outer(smask, smask)
    outside = abs(visc_sp) - abs(visc_sp).multiply(inside)
    print(f"  spdee K(Ek) - K(0) outside the (v,v) and (s,s) blocks: max = {abs(outside).max():.2e} "
          "(viscous and thermal groups are block-separable)")
    ref_max = {(vi, vj): block_max(A_drop, si, bi, bj) for bi, vi in enumerate(VARS)
               for bj, vj in enumerate(VARS)}
    block_table(A0, sum(sign[t] * terms[t] for t in TERMS_INVISCID), si, rows_cls,
                "inviscid group: SPDEE K(Ek=0) vs aligned seed pressure+buoyancy+coriolis+"
                "mass_flux+superadiabatic (rel to the full-operator block max)", ref_max=ref_max)
    block_table(visc_sp, sign["viscous"] * terms["viscous"] + sign["thermal_diff"] * terms["thermal_diff"],
                si, rows_cls, "diffusive group: SPDEE K(Ek) - K(0) vs aligned seed viscous + thermal_diff "
                "(rel to the full-operator block max)", ref_max=ref_max)
    print(f"\n  south-pole closure (SPDEE as solved minus v_theta(pi) = 0): max|S| = {abs(S_pole).max():.3e}, "
          f"nnz = {S_pole.nnz}; rows touched: "
          + ", ".join(sorted({f"{si.var[i]}(j={si.jt[i]})" for i in np.unique(S_pole.nonzero()[0])})))
    k_max = np.unravel_index(abs(S_pole).argmax(), S_pole.shape)
    print(f"  largest entry {S_pole[k_max]:.4f} at row {si.label(k_max[0])} col {si.label(k_max[1])}")

    # Rossby mode of hybrid operators (sparse shift-invert)
    guess = -2.0 / (M + 1) - 0.02j
    om_seed = rossby_sparse(A_seed, guess)
    print("\n  sectoral Rossby mode by shift-invert (Omega0 units), d = difference from the seed:")
    print(f"    seed (strict conversion)               {om_seed:.6f}")
    rows = [("seed, signs aligned to SPDEE", A_al),
            ("spdee, v_theta(pole) = 0", A_drop),
            ("spdee as solved (schur)", A_schur),
            ("aligned seed + SPDEE south-pole S", A_al + S_pole),
            ("aligned seed + SPDEE inviscid group",
             A_al - sum(sign[t] * terms[t] for t in TERMS_INVISCID) + A0),
            ("aligned seed + SPDEE viscous (v,v)",
             A_al - sign["viscous"] * terms["viscous"] + visc_sp.multiply(np.outer(vmask, vmask))),
            ("aligned seed + SPDEE thermal (s,s)",
             A_al - sign["thermal_diff"] * terms["thermal_diff"] + visc_sp.multiply(np.outer(smask, smask)))]
    Hs = (A_drop - A_al).tocsr()
    for cls in ("interior", "wall", "pole"):
        rmask = (rows_cls == cls) & vmask
        rows.append((f"aligned seed + SPDEE (v,v) {cls} rows",
                     A_al + Hs.multiply(np.outer(rmask, vmask))))
    seed_inv = sum(sign[t] * terms[t] for t in TERMS_INVISCID)
    H0 = (A0 - seed_inv).tocsr()
    for bi, vi in enumerate(VARS):
        for bj, vj in enumerate(VARS):
            if block_max(H0, si, bi, bj) > 1e-8 * max(block_max(A0, si, bi, bj), 1e-300):
                bmask = np.outer(si.block == bi, si.block == bj)
                rows.append((f"aligned seed + SPDEE inviscid block {vi},{vj}",
                             A_al + H0.multiply(bmask)))
    for name, A in rows:
        om = rossby_sparse(sps.csr_matrix(A), guess)
        print(f"    {name:42s} {om:.6f}   d = {om - om_seed:+.2e}")

    if args.no_eig:
        return

    # ---- (b) dense spectra in the physical window ----
    print("\n(b) dense eigenvalues in the window |Re| <= 2.2, -0.6 <= Im <= 0.1 (Omega0 units); "
          "pairs by optimal one-to-one assignment against the seed")
    ref_all, t_seed = dense_omega(Ms, seed=True)
    om_ref = ref_all[in_window(ref_all)]
    grow = ref_all[ref_all.imag > 1e-6]
    print(f"  {'seed':36s}: {len(om_ref)} in window ({t_seed:.0f} s); Rossby {rossby_of(om_ref):.5f}; "
          f"growing (Im > 1e-6): {len(grow)}" + (f", max Im = {grow.imag.max():.3e}" if len(grow) else ""))
    solves = [("spdee, v_theta(pole) = 0", A_drop), ("spdee as solved (schur)", A_schur)]
    if args.dense_hybrids:
        solves += [(name, A) for name, A in rows if name.startswith("seed, signs")
                   or "group" in name or "block vp," in name]
    spectra = {}
    for name, A in solves:
        om, t = dense_omega(sps.csr_matrix(A))
        spectra[name] = om
        report_spectrum(name, om, ref_all, t)
    om_d = spectra[solves[0][0]]
    dd, _ = pair(om_d, spectra[solves[1][0]])
    moved = om_d[dd > 1e-6 * np.maximum(1.0, np.abs(om_d))]
    d_win = dd[in_window(om_d)]
    print(f"  south-pole closure moves {len(moved)} of {len(dd)} eigenvalues by more than 1e-6 relative, "
          f"{int(in_window(moved).sum())} of them inside the window (max in-window |d omega| = {d_win.max():.2e})"
          + (f"; movers span Re [{moved.real.min():.3g}, {moved.real.max():.3g}], "
             f"Im [{moved.imag.min():.3g}, {moved.imag.max():.3g}]" if len(moved) else ""))


if __name__ == "__main__":
    main()
