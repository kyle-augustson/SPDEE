"""Classification of an eigenpair of a compiled problem by its structure.

Every diagnostic reads the eigenvector on the physical meridional grid of
its core (``to_grid_fields``): a scalar as f(r, theta), a vector as
(v_r, v_theta, v_phi)(r, theta) with the spin components mapped by
v_theta = (v_+ + v_-)/sqrt2, v_phi = (v_+ - v_-)/(i sqrt2), and the
quadrature weights w_r (including r^2) and w_theta (including sin theta) of
that grid, so that

    E[f] = sum_ij w_r(i) w_theta(j) |f(r_i, theta_j)|^2

is the L2 energy of a field (2 pi times the volume integral of |f|^2, the
kinetic energy of a velocity up to the density).  The staggered core's
face components are averaged to the cell centres first; the sh_fd core is
synthesized on its Gauss-Legendre grid; the Dedalus core is read on its own
grid with the exact quadrature of its nodes.

Quantities (all as fractions of the total energy of the state unless said):

energy fractions   E[component] / E[state] per unknown and per component
                   (r, theta, phi, or the horizontal pair); the fractions
                   over the dynamical unknowns are the kinetic / magnetic /
                   thermal split of the mode.
l-content          per component of spin s the coefficients
                   a_l(r) = 2 pi sum_j w_theta(j) conj(Y^s_lm(theta_j)) f(r_i, theta_j)
                   on the spin-weighted harmonics of the core's m, l = |m|
                   .. |m| + N_theta - 1, and the fractions
                   E_l = sum_i w_r(i) |a_l(r_i)|^2 / sum_l E_l.  Exact on
                   the sh_fd core (its state is the coefficients), a
                   quadrature projection on the finite-difference grids.
toroidal/poloidal  for a vector field, per l, the horizontal spin
                   coefficients split into the spheroidal (poloidal) pattern
                   (a_-, a_+) ~ (k^-, k^+) = (1, -1) and the toroidal pattern
                   r_hat x grad Y, (a_-, a_+) ~ (1, 1):
                   S_l = (a_- - a_+)/sqrt2,  T_l = (a_- + a_+)/sqrt2,
                   E_tor = sum_l int |T_l|^2 r^2 dr,
                   E_pol = sum_l int (|S_l|^2 + |a^0_l|^2) r^2 dr,
                   fractions of E_tor + E_pol.
parity             p = Re <x, R x> / <x, x> in the energy inner product with
                   R the reflection theta -> pi - theta carrying the signs
                   (+, +, -, +) on (scalar, v_r, v_theta, v_phi): p = +1 is
                   equatorially symmetric (Zhang's SYM: pressure even in z),
                   p = -1 antisymmetric.
localisation       radial centroid and width of the energy density,
                   r_bar = int r E(r) dr / int E(r) dr,
                   width = sqrt(int (r - r_bar)^2 E / int E),
                   E(r_i) = w_r(i) sum_j w_theta(j) |x|^2, and the same in
                   theta (a critical layer shows as a small width).
pencil residual    ||K x - lambda M x||_2 / (||K x||_2 + |lambda| ||M x||_2).
constraint residual  per algebraic equation (a row block of the pencil with
                   no mass), ||K_rows x||_2 / ||K x||_2: the discrete
                   divergence of a constrained field, or for a staggered
                   flux unknown its exact flux divergence.

``identify`` consolidates the eigenvector-based identifications: among
the eigenpairs of a solve it picks the one matching a ``ModeTarget`` by
structure (parity, toroidal fraction, dominant l, energy split, a
user-supplied misfit such as a polynomial fit of the pressure) rather
than by the nearest eigenvalue, and reports the discrimination ratio

    median over random control frequencies c of min_j |Re omega_j - c|
    -----------------------------------------------------------------
                   |Re omega_identified - omega_target|

which is of order one when the spectrum is dense enough to match any
frequency (the staggered inertial band, docs/analytic_benchmarks.md) and
large when the identification is sharper than the mode spacing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spdee.backends.sh_fd.angular import sYlm_theta
from spdee.analysis.weights import _trapezoid_weights, _cell_weights

__all__ = ["GridFields", "to_grid_fields", "ModeFeatures", "features", "energy_fractions",
           "l_content", "toroidal_poloidal", "parity", "localisation", "pencil_residual",
           "constraint_residual", "ModeTarget", "Identification", "identify",
           "staggered_node_index", "staggered_full_values"]

SQ2 = np.sqrt(2.0)


# --------------------------------------------------------------- grid data

@dataclass
class GridFields:
    """An eigenvector on the physical meridional grid of its core: scalars
    (1, N_r, N_theta), vectors (3, N_r, N_theta) as (r, theta, phi), tensors
    (9, N_r, N_theta) in the core's component order."""
    r: np.ndarray
    theta: np.ndarray
    w_r: np.ndarray
    w_theta: np.ndarray
    fields: Dict[str, np.ndarray]
    kinds: Dict[str, str]
    m: int

    def energy(self, f: np.ndarray) -> float:
        """sum_ij w_r w_theta |f|^2 summed over the leading component axis."""
        return float(np.einsum("i,j,cij->", self.w_r, self.w_theta, np.abs(f) ** 2))

    def total(self) -> float:
        return sum(self.energy(f) for f in self.fields.values())

    def spin(self, name: str) -> Dict[int, np.ndarray]:
        """Spin components {0, +1, -1} of a vector, {0} of a scalar."""
        f = self.fields[name]
        if f.shape[0] == 1:
            return {0: f[0]}
        if f.shape[0] != 3:
            raise ValueError(f"{name!r} is not a scalar or vector")
        vr, vt, vp = f
        return {0: vr, +1: (vt + 1j * vp) / SQ2, -1: (vt - 1j * vp) / SQ2}


def _legendre_quadrature(nodes: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Weights integrating polynomials of degree < len(nodes) exactly on
    [lo, hi] at the given nodes (a Vandermonde solve in the Legendre basis)."""
    from numpy.polynomial.legendre import legvander
    x = (2.0 * np.asarray(nodes, float) - (lo + hi)) / (hi - lo)
    n = len(x)
    V = legvander(x, n - 1).T
    rhs = np.zeros(n)
    rhs[0] = 2.0
    w = np.linalg.solve(V, rhs)
    return w * 0.5 * (hi - lo)


def to_grid_fields(cp, x: np.ndarray) -> GridFields:
    """The eigenvector ``x`` of ``cp`` on its physical grid (module docstring)."""
    disc = getattr(cp, "discretization", None)
    if disc == "sh_fd":
        return _sh_fd_fields(cp, x)
    if disc == "dedalus":
        return _dedalus_fields(cp, x)
    if disc == "staggered":
        return _staggered_fields(cp, x)
    if hasattr(cp, "matrices"):
        return _collocated_fields(cp, x)
    raise TypeError(f"cannot read eigenvectors of {type(cp).__name__}")


def _spin_to_physical(comp: Dict[int, np.ndarray]) -> np.ndarray:
    vr, vp_, vm = comp[0], comp[+1], comp[-1]
    return np.stack([vr, (vp_ + vm) / SQ2, (vp_ - vm) / (1j * SQ2)])


def _collocated_fields(cp, x) -> GridFields:
    cm = cp.matrices
    grid, layout = cm.grid, cm.layout
    data = layout.unpack(np.asarray(x, dtype=complex))
    fields, kinds = {}, {}
    for name in layout.all_names:
        kind = layout.entry(name).kind
        kinds[name] = kind
        if kind == "scalar":
            fields[name] = data[name][None]
        elif kind == "vector":
            fields[name] = _spin_to_physical(data[name])
        else:
            fields[name] = np.stack([data[name][k] for k in sorted(data[name])])
    th = grid.theta
    return GridFields(r=grid.r, theta=th, w_r=_trapezoid_weights(grid.r) * grid.r ** 2,
                      w_theta=_cell_weights(th, 0.0, np.pi) * np.sin(th),
                      fields=fields, kinds=kinds, m=cm.m)


def _sh_fd_fields(cp, x) -> GridFields:
    theta, wq = cp.ang.quadrature(cp.grid.nquad)
    data = cp.unpack(np.asarray(x, dtype=complex))
    fields, kinds = {}, {}
    for name, kind, _off, _size in cp.layout:
        c = data[name]
        kinds[name] = kind
        if kind == "scalar":
            fields[name] = cp.to_grid(c[0], theta, 0).T[None]
        elif kind == "vector":
            comp = {-1: cp.to_grid(c[0], theta, -1).T, +1: cp.to_grid(c[1], theta, +1).T,
                    0: cp.to_grid(c[2], theta, 0).T}
            fields[name] = _spin_to_physical(comp)
        else:
            spins = (-1, +1, 0)
            fields[name] = np.stack([cp.to_grid(c[3 * a + b], theta, spins[a] + spins[b]).T
                                     for a in range(3) for b in range(3)])
    r = cp.r
    wr = (_cell_weights(r, 0.0, cp.grid.r_outer) if cp.ball else _trapezoid_weights(r)) * r ** 2
    return GridFields(r=r, theta=theta, w_r=wr, w_theta=np.asarray(wq, float),
                      fields=fields, kinds=kinds, m=cp.m)


def _dedalus_fields(cp, x) -> GridFields:
    data = cp.eigenvector_fields(np.asarray(x, dtype=complex))
    fields, kinds = {}, {}
    for name, arr in data.items():
        arr = np.asarray(arr)
        if arr.ndim == 2:
            kinds[name] = "scalar"
            fields[name] = arr.T[None]
        elif arr.ndim == 3:
            kinds[name] = "vector"
            fields[name] = np.stack([arr[2].T, arr[1].T, arr[0].T])
        else:
            kinds[name] = "tensor"
            fields[name] = arr.reshape(9, *arr.shape[-2:]).transpose(0, 2, 1)
    r, th = np.asarray(cp.r_grid, float), np.asarray(cp.theta_grid, float)
    g = cp.grid
    return GridFields(r=r, theta=th, w_r=_legendre_quadrature(r, g.r_inner, g.r_outer) * r ** 2,
                      w_theta=_legendre_quadrature(np.cos(th), -1.0, 1.0),
                      fields=fields, kinds=kinds, m=cp.m)


def staggered_node_index(cp, fld) -> np.ndarray:
    """Full sub-grid node index of every reduced DOF of a staggered field:
    the wall rings of a normal component dropped by the symmetric DOF
    reduction are absent (outer ring first, then the inner or origin ring,
    or the folded origin slaves of m = 0, |m| = 1 on a ball)."""
    g = cp.grids
    Nt_c = len(g.theta_center)
    sizes = {"center": len(g.r_center) * Nt_c, "r_edge": len(g.r_edge) * Nt_c,
             "theta_edge": len(g.r_center) * len(g.theta_edge),
             "corner": len(g.r_edge) * len(g.theta_edge)}
    full = sizes[fld.location]
    keep = np.ones(full, dtype=bool)
    deficit = full - fld.size
    if deficit > 0 and fld.location == "r_edge":
        Nr_e = len(g.r_edge)
        if deficit >= Nt_c:
            keep[(Nr_e - 1) * Nt_c:] = False
            deficit -= Nt_c
        if deficit == Nt_c:
            keep[:Nt_c] = False
        elif deficit == Nt_c - 1:
            keep[1:Nt_c] = False
        elif deficit:
            raise ValueError(f"cannot place the {deficit} dropped nodes of {fld.name!r}")
    elif deficit:
        raise ValueError(f"unexpected DOF reduction on {fld.name!r} ({fld.location})")
    return np.nonzero(keep)[0]


def staggered_full_values(cp, fld, xf: np.ndarray) -> np.ndarray:
    """The field on its full sub-grid, dropped rings zero and folded origin
    slaves restored from their master; shape (N_r_loc, N_theta_loc)."""
    g = cp.grids
    Nt_c = len(g.theta_center)
    nodes = {"center": (len(g.r_center), Nt_c), "r_edge": (len(g.r_edge), Nt_c),
             "theta_edge": (len(g.r_center), len(g.theta_edge)),
             "corner": (len(g.r_edge), len(g.theta_edge))}
    shape = nodes[fld.location]
    out = np.zeros(shape[0] * shape[1], dtype=complex)
    idx = staggered_node_index(cp, fld)
    out[idx] = xf
    if fld.location == "r_edge" and shape[0] * shape[1] - fld.size in (Nt_c - 1, 2 * Nt_c - 1):
        profile = np.cos(g.theta_center) if cp.m == 0 else np.sin(g.theta_center)
        out[1:Nt_c] = out[0] * profile[1:] / profile[0]
    return out.reshape(shape)


def _staggered_fields(cp, x) -> GridFields:
    g = cp.grids
    x = np.asarray(x, dtype=complex)
    rc, tc = g.r_center, g.theta_center
    full = {f.name: staggered_full_values(cp, f, x[f.slice]) for f in cp.layout.fields}
    fields, kinds = {}, {}
    placements = cp.layout.placements or {}
    for u, placement in placements.items():
        if placement == "face":
            vr = 0.5 * (full[u + "_r"][1:] + full[u + "_r"][:-1])
            vt = 0.5 * (full[u + "_t"][:, 1:] + full[u + "_t"][:, :-1])
            fields[u] = np.stack([vr, vt, full[u + "_p"]])
            kinds[u] = "vector"
        elif placement == "center":
            fields[u] = full[u][None]
            kinds[u] = "scalar"
        else:
            comps = [full[f.name] for f in cp.layout.fields if f.name.startswith(u + "_")]
            fields[u] = np.stack([_to_centres(c, len(rc), len(tc)) for c in comps])
            kinds[u] = "tensor"
    hr = g.r_edge[1] - g.r_edge[0]
    ht = tc[1] - tc[0] if len(tc) > 1 else np.pi
    return GridFields(r=rc, theta=tc, w_r=np.full(len(rc), hr) * rc ** 2,
                      w_theta=np.full(len(tc), ht) * np.sin(tc), fields=fields, kinds=kinds, m=cp.m)


def _to_centres(a: np.ndarray, nr: int, nt: int) -> np.ndarray:
    if a.shape[0] == nr + 1:
        a = 0.5 * (a[1:] + a[:-1])
    if a.shape[1] == nt + 1:
        a = 0.5 * (a[:, 1:] + a[:, :-1])
    return a


# --------------------------------------------------------------- features

def energy_fractions(gf: GridFields) -> Dict[str, Dict[str, float]]:
    """{unknown: {'total': f, 'r': f, 'theta': f, 'phi': f, 'horizontal': f}}
    (vectors) or {'total': f} (scalars), fractions of the state's energy."""
    tot = gf.total() or 1.0
    out = {}
    for name, f in gf.fields.items():
        d = {"total": gf.energy(f) / tot}
        if f.shape[0] == 3:
            e = [gf.energy(f[k:k + 1]) / tot for k in range(3)]
            d.update(r=e[0], theta=e[1], phi=e[2], horizontal=e[1] + e[2])
        elif f.shape[0] == 9:
            d.update({f"c{k}": gf.energy(f[k:k + 1]) / tot for k in range(9)})
        out[name] = d
    return out


def _harmonics(gf: GridFields, s: int, l_max: Optional[int] = None):
    m = abs(gf.m)
    if l_max is None:
        l_max = m + len(gf.theta) - 1
    ls = np.arange(max(m, abs(s)), l_max + 1)
    Y = np.array([sYlm_theta(s, l, m, gf.theta) for l in ls])
    return ls, Y


def _project(gf: GridFields, f: np.ndarray, s: int, l_max=None) -> Tuple[np.ndarray, np.ndarray]:
    """Coefficients a_l(r_i) of one component of spin s, shape (nl, N_r)."""
    ls, Y = _harmonics(gf, s, l_max)
    a = 2 * np.pi * (np.conj(Y) * gf.w_theta[None, :]) @ f.T
    return ls, a


def l_content(gf: GridFields, name: str, l_max: Optional[int] = None):
    """(ls, fractions): the energy of ``name`` per spherical-harmonic degree,
    summed over its spin components, as fractions of its projected energy."""
    m = abs(gf.m)
    if l_max is None:
        l_max = m + len(gf.theta) - 1
    ls_out = np.arange(m, l_max + 1)
    E = np.zeros(len(ls_out))
    for s, f in gf.spin(name).items():
        ls, a = _project(gf, f, s, l_max)
        E[np.searchsorted(ls_out, ls)] += np.einsum("li,i->l", np.abs(a) ** 2, gf.w_r)
    return ls_out, E / (E.sum() or 1.0)


def toroidal_poloidal(gf: GridFields, name: str, l_max: Optional[int] = None) -> Dict[str, object]:
    """Toroidal and poloidal energy fractions of the vector ``name`` and the
    per-l split (module docstring)."""
    comps = gf.spin(name)
    if len(comps) != 3:
        raise ValueError(f"{name!r} is not a vector")
    ls0, a0 = _project(gf, comps[0], 0, l_max)
    lsp, ap = _project(gf, comps[+1], +1, l_max)
    lsm, am = _project(gf, comps[-1], -1, l_max)
    ls = ls0
    ap_full = np.zeros_like(a0)
    am_full = np.zeros_like(a0)
    ap_full[np.searchsorted(ls, lsp)] = ap
    am_full[np.searchsorted(ls, lsm)] = am
    S = (am_full - ap_full) / SQ2
    T = (am_full + ap_full) / SQ2
    e_tor = np.einsum("li,i->l", np.abs(T) ** 2, gf.w_r)
    e_pol = np.einsum("li,i->l", np.abs(S) ** 2 + np.abs(a0) ** 2, gf.w_r)
    tot = e_tor.sum() + e_pol.sum() or 1.0
    return {"toroidal": float(e_tor.sum() / tot), "poloidal": float(e_pol.sum() / tot),
            "ls": ls, "toroidal_l": e_tor / tot, "poloidal_l": e_pol / tot}


def parity(gf: GridFields, name: Optional[str] = None) -> float:
    """Equatorial parity in [-1, 1] of one unknown or of the whole state
    (+1 symmetric, -1 antisymmetric)."""
    names = [name] if name is not None else list(gf.fields)
    num = den = 0.0
    for n in names:
        f = gf.fields[n]
        if f.shape[0] == 3:
            signs = np.array([1.0, -1.0, 1.0])
        elif f.shape[0] == 1:
            signs = np.array([1.0])
        else:
            continue
        Rf = f[:, :, ::-1] * signs[:, None, None]
        num += float(np.real(np.einsum("i,j,cij->", gf.w_r, gf.w_theta, np.conj(f) * Rf)))
        den += gf.energy(f)
    return num / (den or 1.0)


def localisation(gf: GridFields, name: Optional[str] = None) -> Dict[str, float]:
    """Radial and latitudinal centroid and width of the energy density."""
    names = [name] if name is not None else list(gf.fields)
    dens = sum(np.einsum("cij->ij", np.abs(gf.fields[n]) ** 2) for n in names)
    Er = gf.w_r * (dens @ gf.w_theta)
    Et = gf.w_theta * (gf.w_r @ dens)
    tot = Er.sum() or 1.0
    rbar = float(Er @ gf.r / tot)
    tbar = float(Et @ gf.theta / tot)
    return {"r_centroid": rbar, "r_width": float(np.sqrt(Er @ (gf.r - rbar) ** 2 / tot)),
            "theta_centroid": tbar,
            "theta_width": float(np.sqrt(Et @ (gf.theta - tbar) ** 2 / tot)),
            "latitude_width": float(np.sqrt(Et @ (gf.theta - np.pi / 2) ** 2 / tot))}


def pencil_residual(K, M, lam, V) -> np.ndarray:
    """Relative pencil residual per column of V (M None for a standard problem)."""
    V = np.asarray(V, dtype=complex)
    if V.ndim == 1:
        V = V[:, None]
    lam = np.atleast_1d(np.asarray(lam, dtype=complex))
    KV = K @ V
    MV = V if M is None else M @ V
    num = np.linalg.norm(KV - MV * lam[None, :], axis=0)
    den = np.linalg.norm(KV, axis=0) + np.abs(lam) * np.linalg.norm(MV, axis=0)
    return num / np.where(den == 0, 1.0, den)


def constraint_residual(cp, x, K=None, M=None) -> Dict[str, float]:
    """||K_rows x|| / ||K x|| over the rows of each algebraic equation (zero
    rows of M), plus the exact flux divergence of a staggered involution."""
    if K is None or M is None:
        K, M = cp.assemble_evp_matrices()
    x = np.asarray(x, dtype=complex)
    Kx = K @ x
    nK = np.linalg.norm(Kx) or 1.0
    massless = np.asarray(abs(M).sum(axis=1)).ravel() == 0
    out = {}
    slices = _equation_rows(cp, K.shape[0])
    for eq, rows in slices.items():
        sel = np.zeros(K.shape[0], dtype=bool)
        sel[rows] = True
        sel &= massless
        if sel.any():
            out[eq] = float(np.linalg.norm(Kx[sel]) / nK)
    if not slices:
        out["massless_rows"] = float(np.linalg.norm(Kx[massless]) / nK)
    div_ops = ((getattr(cp, "constraints", None) or {}).get("divergence") or {})
    for b in div_ops:
        out[f"div({b})"] = float(np.linalg.norm(cp.divergence(x, b)) / (np.linalg.norm(x) or 1.0))
    return out


def _equation_rows(cp, N) -> Dict[str, np.ndarray]:
    if hasattr(cp, "matrices"):
        return {eq: np.arange(s.start, s.stop) for eq, s in cp.matrices.equation_row_slices.items()}
    if getattr(cp, "discretization", None) == "sh_fd":
        offs = sorted(cp._eq_row_offset.items(), key=lambda kv: kv[1])
        out = {}
        for i, (eq, start) in enumerate(offs):
            stop = offs[i + 1][1] if i + 1 < len(offs) else N
            out[eq] = np.arange(start, stop)
        return out
    return {}


@dataclass
class ModeFeatures:
    """The structural description of one eigenpair."""
    lam: complex
    omega: complex
    energy: Dict[str, Dict[str, float]]
    parity: float
    localisation: Dict[str, float]
    toroidal: Optional[float] = None
    poloidal: Optional[float] = None
    l_dominant: Optional[int] = None
    l_fractions: Optional[Tuple[np.ndarray, np.ndarray]] = None
    residual: Optional[float] = None
    vector: Optional[str] = None

    def summary(self) -> str:
        parts = [f"omega={self.omega:.6g}", f"parity={self.parity:+.3f}"]
        if self.toroidal is not None:
            parts.append(f"toroidal={self.toroidal:.3f}")
        if self.l_dominant is not None:
            parts.append(f"l={self.l_dominant}")
        e = ", ".join(f"{k}={v['total']:.3f}" for k, v in self.energy.items())
        parts.append(f"energy[{e}]")
        parts.append(f"r={self.localisation['r_centroid']:.3f}+-{self.localisation['r_width']:.3f}")
        if self.residual is not None:
            parts.append(f"resid={self.residual:.1e}")
        return " ".join(parts)


def features(cp, lam: complex, x: np.ndarray, vector: Optional[str] = None,
             K=None, M=None, l_max: Optional[int] = None) -> ModeFeatures:
    """All diagnostics of one eigenpair; ``vector`` names the vector unknown
    of the toroidal/poloidal split and the l-content (default: the first)."""
    gf = to_grid_fields(cp, x)
    vecs = [n for n, k in gf.kinds.items() if k == "vector"]
    if vector is None and vecs:
        vector = vecs[0]
    feat = ModeFeatures(lam=complex(lam), omega=complex(-1j * lam), energy=energy_fractions(gf),
                        parity=parity(gf), localisation=localisation(gf), vector=vector)
    if vector is not None:
        tp = toroidal_poloidal(gf, vector, l_max)
        feat.toroidal, feat.poloidal = tp["toroidal"], tp["poloidal"]
        ls, fr = l_content(gf, vector, l_max)
        feat.l_fractions = (ls, fr)
        feat.l_dominant = int(ls[np.argmax(fr)])
    elif gf.fields:
        name = next(iter(gf.fields))
        ls, fr = l_content(gf, name, l_max)
        feat.l_fractions = (ls, fr)
        feat.l_dominant = int(ls[np.argmax(fr)])
    if K is not None:
        feat.residual = float(pencil_residual(K, M, lam, x)[0])
    return feat


# ------------------------------------------------------------- identify

@dataclass
class ModeTarget:
    """What to look for.  ``omega`` is the target frequency (omega = -i lambda,
    its real part is matched); ``window`` restricts the candidates to
    |Re omega - omega| < window; the structural gates are optional; ``score``
    is a misfit of a ``ModeFeatures`` and its eigenvector (smaller is better)
    that decides among the survivors, else the nearest frequency does."""
    omega: Optional[float] = None
    window: Optional[float] = None
    parity: Optional[str] = None
    toroidal_min: Optional[float] = None
    poloidal_min: Optional[float] = None
    l: Optional[int] = None
    vector: Optional[str] = None
    energy: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    neutral: Optional[float] = None
    score: Optional[Callable[[ModeFeatures, np.ndarray], float]] = None

    def passes(self, f: ModeFeatures) -> List[str]:
        """The gates the features fail (empty when the mode qualifies)."""
        fails = []
        if self.parity == "SYM" and f.parity < 0:
            fails.append("parity")
        if self.parity == "ANTI" and f.parity > 0:
            fails.append("parity")
        if self.toroidal_min is not None and (f.toroidal or 0.0) < self.toroidal_min:
            fails.append("toroidal")
        if self.poloidal_min is not None and (f.poloidal or 0.0) < self.poloidal_min:
            fails.append("poloidal")
        if self.l is not None and f.l_dominant != self.l:
            fails.append("l")
        for name, (lo, hi) in self.energy.items():
            e = f.energy.get(name, {}).get("total", 0.0)
            if not lo <= e <= hi:
                fails.append(f"energy[{name}]")
        if self.neutral is not None and abs(f.lam.real) > self.neutral:
            fails.append("neutral")
        return fails


@dataclass
class Identification:
    index: Optional[int]
    lam: Optional[complex]
    omega: Optional[complex]
    features: Optional[ModeFeatures]
    misfit: Optional[float]
    nearest_index: Optional[int]
    discrimination: Optional[float]
    control_spacing: Optional[float]
    candidates: List[Tuple[int, List[str], Optional[float]]]

    @property
    def differs_from_nearest(self) -> bool:
        return self.index is not None and self.index != self.nearest_index

    def __str__(self) -> str:
        if self.index is None:
            return "identify: no eigenpair passes the gates"
        s = (f"identify: index {self.index}, omega = {self.omega:.6g}"
             f" (nearest-eigenvalue index {self.nearest_index})")
        if self.misfit is not None:
            s += f", misfit {self.misfit:.3g}"
        if self.discrimination is not None:
            s += (f", discrimination {self.discrimination:.3g}"
                  f" (control spacing {self.control_spacing:.3g})")
        return s + "\n  " + self.features.summary()


def identify(cp, lam: Sequence[complex], V: np.ndarray, target: ModeTarget,
             K=None, M=None, n_controls: int = 40, seed: int = 3,
             l_max: Optional[int] = None) -> Identification:
    """Pick the eigenpair of (lam, V) matching ``target`` by structure."""
    lam = np.asarray(lam, dtype=complex)
    V = np.asarray(V, dtype=complex)
    om = -1j * lam
    finite = np.isfinite(lam)
    cand = np.nonzero(finite)[0]
    nearest = None
    if target.omega is not None:
        dist = np.where(finite, np.abs(om.real - target.omega), np.inf)
        nearest = int(np.argmin(dist)) if finite.any() else None
        if target.window is not None:
            cand = cand[dist[cand] < target.window]
    records = []
    best = None
    feats = {}
    for j in cand:
        f = features(cp, lam[j], V[:, j], vector=target.vector, K=K, M=M, l_max=l_max)
        feats[j] = f
        fails = target.passes(f)
        score = None
        if not fails:
            if target.score is not None:
                score = float(target.score(f, V[:, j]))
            elif target.omega is not None:
                score = float(abs(om[j].real - target.omega))
            else:
                score = 0.0
            if best is None or score < best[1]:
                best = (j, score)
        records.append((int(j), fails, score))
    disc = spacing = None
    if target.omega is not None and finite.sum() > 1:
        band = om.real[finite]
        rng = np.random.default_rng(seed)
        controls = rng.uniform(band.min(), band.max(), n_controls)
        spacing = float(np.median([np.abs(band - c).min() for c in controls]))
    if best is None:
        return Identification(None, None, None, None, None, nearest, None, spacing, records)
    j, score = best
    if spacing is not None:
        err = abs(om[j].real - target.omega)
        disc = float(spacing / err) if err > 0 else np.inf
    return Identification(int(j), complex(lam[j]), complex(om[j]), feats[j], score, nearest,
                          disc, spacing, records)
