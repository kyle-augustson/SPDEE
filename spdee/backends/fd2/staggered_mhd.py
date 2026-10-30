"""Constrained-transport MHD on the Arakawa-C grid.

The magnetic state is stored as face fluxes, so div(b) = 0 is an exact
discrete identity:

    Phi_r   on r-edges      (r_edge,   theta_center)   (Nr+1, Nt)
    Phi_th  on theta-edges  (r_center, theta_edge)     (Nr,   Nt+1)
    Phi_ph  on cells        (r_center, theta_center)   (Nr,   Nt)

The EMF lives on edges as circulations (E_phi on corners, E_r on
theta-edges, E_th on r-edges) and Faraday's law is the incidence map from
circulations to flux tendencies.  The physical field b = Phi / area enters
only the Lorentz force and the resistive EMF.

On a ball (r_edge[0] = 0) the origin ring of Phi_r crosses faces of zero
area and the circulations on the origin's theta- and phi-arcs have zero
length, so the ring is identically zero with zero tendency and is
pinned by the assembler, the involution Div Curl = 0 holds cell by cell
with no closure value, and the only origin closure is the first-centre
radial average of the Lorentz force, which reads its origin node by the
vector origin parity (docs/ball_staggered.md section 7).

The system

    induction:  Dt(b) + c_ind * Curl(Cross(v, B0)) - eta * Lap(b) = 0
    momentum:   ... + c1 * f * Cross(Curl(b), B0) + c2 * g * Cross(J0, b)

with B0, J0 background vectors and f, g scalar coefficients is dispatched
task by task by ``staggered_general``, which builds the blocks of
``ct_induction_blocks``, ``ct_resistive_blocks``, ``ct_lorentz_blocks``
and ``ct_current_blocks`` on the placement of section 2 of
docs/staggered_general_design.md.  Wall conditions on b: 'conducting'
(Dot(r_hat, b) = 0 and the tangential electric field zero), 'radial'
(Dot(r_hat, b) = 0 only) or 'free'.  Pinned tangential components cannot
be imposed on face fluxes and are rejected.

Blocks are built in the ``dt x = A_phys x`` convention and negated by the
caller, matching the staggered assembler's ``A = -A_phys``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

__all__ = ["CTGrid", "ct_induction_blocks", "ct_resistive_blocks",
           "ct_lorentz_blocks", "ct_current_blocks"]


def _D(n):
    """Raw difference, faces (n+1) -> centers (n)."""
    return sp.diags([-np.ones(n), np.ones(n)], [0, 1], shape=(n, n + 1))


def _dd(a):
    return sp.diags(np.asarray(a, dtype=complex).ravel())


def _safe_inv(a):
    """1/a with zero where a vanishes (pole faces), relative to the median."""
    a = np.asarray(a, float).ravel()
    out = np.zeros_like(a)
    scale = np.median(np.abs(a[np.abs(a) > 0])) if np.any(np.abs(a) > 0) else 1.0
    nz = np.abs(a) > 1e-9 * scale
    out[nz] = 1.0 / a[nz]
    return out


class CTGrid:
    """Geometry, incidence operators, velocity/edge interpolations and the dual
    curl of the (r, theta) C-grid, from a spdee ``StaggeredGrids``.  Indices are
    C-ordered with r outer and theta inner.

    ``pole_closure`` closes j_r on the pole theta-edges at m = 0, where
    (1/(r sin theta)) d_theta(sin theta b_phi) has the finite limit
    N'(0) / r with N odd about the pole: N(theta_1) / (r h_theta), second
    order; without it the pole edge carries no current (exact for |m| >= 1,
    where regularity makes j_r vanish at the pole, an O(1) error at m = 0).

    On a ball the r-averages onto the first centre (``K_to_vth``,
    ``rf_to_c``) read their origin node -- a zero-area face, a
    zero-length edge, a corner where 1 / r is guarded -- as the parity
    reflection of the first interior node, ``_origin_average``; every
    other block is the shell's, the degenerate circulations vanishing by
    length."""

    def __init__(self, grids, m: int, pole_closure: bool = False):
        Rf, rc = np.asarray(grids.r_edge, float), np.asarray(grids.r_center, float)
        Tf, tc = np.asarray(grids.theta_edge, float), np.asarray(grids.theta_center, float)
        Nr, Nt = len(rc), len(tc)
        if len(Rf) != Nr + 1 or len(Tf) != Nt + 1:
            raise ValueError("CTGrid expects Nr+1 r-edges and Nt+1 theta-edges")
        self.m, self.Nr, self.Nt = int(m), Nr, Nt
        self.Rf, self.rc, self.Tf, self.tc = Rf, rc, Tf, tc
        self.is_ball = bool(abs(Rf[0]) < 1e-12)
        dr = Rf[1] - Rf[0]
        dth = Tf[1] - Tf[0]
        self.dr, self.dth = dr, dth

        self.RfR, self.tcR = np.meshgrid(Rf, tc, indexing="ij")
        self.rcT, self.TfT = np.meshgrid(rc, Tf, indexing="ij")
        self.rcC, self.tcC = np.meshgrid(rc, tc, indexing="ij")
        self.RfK, self.TfK = np.meshgrid(Rf, Tf, indexing="ij")
        self.RintR, self.tcR_i = np.meshgrid(Rf[1:-1], tc, indexing="ij")
        self.rcT_i, self.TintT = np.meshgrid(rc, Tf[1:-1], indexing="ij")

        self.Area_r = self.RfR ** 2 * np.sin(self.tcR) * dth
        self.Area_th = self.rcT * np.sin(self.TfT) * dr
        self.Area_ph = self.rcC * dr * dth
        self.CellVol = self.rcC ** 2 * np.sin(self.tcC) * dr * dth
        self.Len_ph = self.RfK * np.sin(self.TfK)
        self.Len_r = np.ones((Nr, Nt + 1)) * dr
        self.Len_th = self.RfR * dth

        Ir, It = sp.identity(Nr), sp.identity(Nt)
        Irf, Itf = sp.identity(Nr + 1), sp.identity(Nt + 1)
        self.dthK_rf = sp.kron(Irf, _D(Nt))
        self.drK_tf = sp.kron(_D(Nr), Itf)
        self.dr_rfc = sp.kron(_D(Nr), It)
        self.dth_tfc = sp.kron(Ir, _D(Nt))
        self.div_r, self.div_th = self.dr_rfc, self.dth_tfc

        self.Lph, self.Lr, self.Lth = _dd(self.Len_ph), _dd(self.Len_r), _dd(self.Len_th)
        self.iAr = _dd(_safe_inv(self.Area_r))
        self.iAth = _dd(_safe_inv(self.Area_th))
        self.iAph = _dd(_safe_inv(self.Area_ph))
        self.n_br = (Nr + 1) * Nt
        self.n_bth = Nr * (Nt + 1)
        self.n_bph = Nr * Nt
        self.n_K = (Nr + 1) * (Nt + 1)
        self.pole_closure = bool(pole_closure)
        self._build_interp()._build_curl_b()

    def divergence(self) -> sp.spmatrix:
        """(Phi_r, Phi_th, Phi_ph) -> net flux out of each cell."""
        return sp.hstack([self.div_r, self.div_th, 1j * self.m * sp.identity(self.n_bph)]).tocsr()

    def flux_to_physical(self):
        return self.iAr, self.iAth, self.iAph

    def _build_interp(self):
        Nr, Nt = self.Nr, self.Nt

        def embed_int(n):
            return sp.vstack([sp.csr_matrix((1, n - 1)), sp.identity(n - 1),
                              sp.csr_matrix((1, n - 1))]).tocsr()

        def c2f(n):
            M = sp.lil_matrix((n + 1, n))
            for j in range(1, n):
                M[j, j - 1] = 0.5
                M[j, j] = 0.5
            M[0, 0] = 1.0
            M[n, n - 1] = 1.0
            return M.tocsr()

        Ir, It = sp.identity(Nr), sp.identity(Nt)
        self.vr_to_K = sp.kron(embed_int(Nr), c2f(Nt))
        self.vth_to_K = sp.kron(c2f(Nr), embed_int(Nt))
        self.vphi_to_tf = sp.kron(Ir, c2f(Nt))
        self.vphi_to_rf = sp.kron(c2f(Nr), It)
        self.vth_to_tf = sp.kron(Ir, embed_int(Nt))
        self.vr_to_rf = sp.kron(embed_int(Nr), It)
        selR_int = sp.diags([np.ones(Nr - 1)], [1], shape=(Nr - 1, Nr + 1))
        selT_int = sp.diags([np.ones(Nt - 1)], [1], shape=(Nt - 1, Nt + 1))
        avgT_f2c = 0.5 * sp.diags([np.ones(Nt), np.ones(Nt)], [0, 1], shape=(Nt, Nt + 1))
        avgR_f2c = self._origin_average(
            0.5 * sp.diags([np.ones(Nr), np.ones(Nr)], [0, 1], shape=(Nr, Nr + 1)))
        self.K_to_vr = sp.kron(selR_int, avgT_f2c)
        self.K_to_vth = sp.kron(avgR_f2c, selT_int)
        self.tf_to_c = sp.kron(Ir, avgT_f2c)
        self.rf_to_c = sp.kron(avgR_f2c, It)
        self.jth_to_vr = sp.kron(selR_int, It)
        self.jr_to_vth = sp.kron(Ir, selT_int)
        avgR_c2f = sp.diags([0.5 * np.ones(Nr - 1), 0.5 * np.ones(Nr - 1)], [0, 1],
                            shape=(Nr - 1, Nr))
        avgT_c2f = sp.diags([0.5 * np.ones(Nt - 1), 0.5 * np.ones(Nt - 1)], [0, 1],
                            shape=(Nt - 1, Nt))
        self.c_to_vr = sp.kron(avgR_c2f, It)
        self.c_to_vth = sp.kron(Ir, avgT_c2f)
        self.S_vr = sp.kron(selR_int, It)
        self.S_vt = sp.kron(Ir, selT_int)
        return self

    def _origin_average(self, avg):
        """The r-edge to centre average with its first row closed at the
        ball origin: the origin node is the reflection of the first
        interior node with the vector origin parity p = (-1)^max(|m|-1, 0)
        (a regular field is finite at the origin for p = +1 and vanishes
        there for p = -1), so the two-point interpolation to r_c0 = h / 2
        on the extended nodes {-h, h} reads (3 + p) / 4 of the node at h:
        the node itself for m = 0 and |m| = 1, half of it otherwise (the
        guard's zero average, exact to O(h^3) on a field proportional to
        r).  A shell is untouched."""
        if not self.is_ball:
            return avg
        p = (-1) ** (max(abs(self.m) - 1, 0) % 2)
        avg = avg.tolil()
        avg.rows[0] = [1]
        avg.data[0] = [0.25 * (3 + p)]
        return avg.tocsr()

    def origin_ring(self):
        """Indices of the origin Phi_r DOFs within the magnetic block on a
        ball (zero-area faces, pinned), empty on a shell."""
        return np.arange(self.Nt) if self.is_ball else np.empty(0, dtype=int)

    def _build_curl_b(self):
        """Physical b on faces -> j = curl b on edges (j_phi corners, j_r
        theta-edges, j_th r-edges)."""
        Nr, Nt, m = self.Nr, self.Nt, self.m
        dr, dth = self.dr, self.dth

        def c2f_diff(n, h):
            M = sp.lil_matrix((n + 1, n))
            for j in range(1, n):
                M[j, j - 1] = -1.0 / h
                M[j, j] = 1.0 / h
            return M.tocsr()

        Ir, It = sp.identity(Nr), sp.identity(Nt)
        Irf, Itf = sp.identity(Nr + 1), sp.identity(Nt + 1)
        RfK, rcT, TfT, RfR, tcR, tcC = self.RfK, self.rcT, self.TfT, self.RfR, self.tcR, self.tcC
        self.jphi_bth = _dd(_safe_inv(RfK)) @ sp.kron(c2f_diff(Nr, dr), Itf) @ _dd(rcT.ravel())
        self.jphi_br = -_dd(_safe_inv(RfK)) @ sp.kron(Irf, c2f_diff(Nt, dth))
        N_bphi = sp.kron(Ir, c2f_diff(Nt, dth)) @ _dd(np.sin(tcC).ravel())
        self.jr_bphi = _dd(_safe_inv(rcT * np.sin(TfT))) @ N_bphi
        if self.pole_closure and m == 0:
            P = sp.lil_matrix((Nr * (Nt + 1), Nr * (Nt + 1)))
            for i in range(Nr):
                w = 1.0 / (rcT[i, 0] * dth)
                P[i * (Nt + 1), i * (Nt + 1) + 1] = w
                P[i * (Nt + 1) + Nt, i * (Nt + 1) + Nt - 1] = w
            self.jr_bphi = self.jr_bphi + P.tocsr() @ N_bphi
        self.jr_bth = _dd(-1j * m * _safe_inv(rcT * np.sin(TfT)))
        self.jth_br = _dd(1j * m * _safe_inv(RfR * np.sin(tcR)))
        self.jth_bphi = -_dd(_safe_inv(RfR)) @ sp.kron(c2f_diff(Nr, dr), It) @ _dd(self.rcC.ravel())
        return self

    def wall_masks(self):
        """Masks (1 interior, 0 wall) for E_th on r-edges and E_phi on corners,
        and the indices of the wall Phi_r DOFs within the magnetic block."""
        Nr, Nt = self.Nr, self.Nt
        rf_wall = np.concatenate([np.arange(Nt), np.arange(Nr * Nt, (Nr + 1) * Nt)])
        K_wall = np.concatenate([np.arange(Nt + 1), np.arange(Nr * (Nt + 1), (Nr + 1) * (Nt + 1))])
        m_rf = np.ones(self.n_br); m_rf[rf_wall] = 0.0
        m_K = np.ones(self.n_K); m_K[K_wall] = 0.0
        return _dd(m_rf), _dd(m_K), rf_wall


def _vec_comp(field, comp: str, R, T):
    return np.asarray(field.components[comp].evaluate_on_grid(R, T), dtype=complex) + 0.0 * T


def _add(blocks, i, j, A):
    if A is None:
        return
    blocks[(i, j)] = A if (i, j) not in blocks else blocks[(i, j)] + A


def ct_induction_blocks(ct: CTGrid, B0, cE, conducting: bool
                        ) -> Dict[Tuple[str, str], sp.spmatrix]:
    """dt Phi = -curl(circulations of E), E = cE (v x B0) on the edges (v
    interpolated from the full-face velocity DOFs, B0 at the edge), the
    wall circulations masked for a conducting wall; ``A_phys`` blocks
    keyed (flux, velocity)."""
    m = ct.m
    blocks: Dict[Tuple[str, str], sp.spmatrix] = {}
    B0r_K = _vec_comp(B0, "v_r", ct.RfK, ct.TfK)
    B0t_K = _vec_comp(B0, "v_theta", ct.RfK, ct.TfK)
    B0t_tf = _vec_comp(B0, "v_theta", ct.rcT, ct.TfT)
    B0p_tf = _vec_comp(B0, "v_phi", ct.rcT, ct.TfT)
    B0r_rf = _vec_comp(B0, "v_r", ct.RfR, ct.tcR)
    B0p_rf = _vec_comp(B0, "v_phi", ct.RfR, ct.tcR)
    mask_rf, mask_K, _ = ct.wall_masks()
    S_vr, S_vt = ct.S_vr, ct.S_vt
    Lph, Lr, Lth = ct.Lph, ct.Lr, ct.Lth
    Eph_vr = _dd(cE * B0t_K) @ ct.vr_to_K
    Eph_vth = _dd(-cE * B0r_K) @ ct.vth_to_K
    Er_vth = _dd(cE * B0p_tf) @ ct.vth_to_tf
    Er_vphi = _dd(-cE * B0t_tf) @ ct.vphi_to_tf
    Eth_vphi = _dd(cE * B0r_rf) @ ct.vphi_to_rf
    Eth_vr = _dd(-cE * B0p_rf) @ ct.vr_to_rf
    Cph_vr, Cph_vth = Lph @ Eph_vr, Lph @ Eph_vth
    Cr_vth, Cr_vphi = Lr @ Er_vth, Lr @ Er_vphi
    Cth_vphi, Cth_vr = Lth @ Eth_vphi, Lth @ Eth_vr
    if conducting:
        Cph_vr, Cph_vth = mask_K @ Cph_vr, mask_K @ Cph_vth
        Cth_vphi, Cth_vr = mask_rf @ Cth_vphi, mask_rf @ Cth_vr
    _add(blocks, "br", "vr", (-(ct.dthK_rf @ Cph_vr) + 1j * m * Cth_vr) @ S_vr)
    _add(blocks, "br", "vt", (-(ct.dthK_rf @ Cph_vth)) @ S_vt)
    _add(blocks, "br", "vp", 1j * m * Cth_vphi)
    _add(blocks, "bt", "vt", (-(1j * m * Cr_vth) + (ct.drK_tf @ Cph_vth)) @ S_vt)
    _add(blocks, "bt", "vp", -(1j * m * Cr_vphi))
    _add(blocks, "bt", "vr", (ct.drK_tf @ Cph_vr) @ S_vr)
    _add(blocks, "bp", "vp", -(ct.dr_rfc @ Cth_vphi) + (ct.dth_tfc @ Cr_vphi))
    _add(blocks, "bp", "vr", (-(ct.dr_rfc @ Cth_vr)) @ S_vr)
    _add(blocks, "bp", "vt", (ct.dth_tfc @ Cr_vth) @ S_vt)
    return blocks


def ct_mean_flow_emf_blocks(ct: CTGrid, U0, cE, conducting: bool
                            ) -> Dict[Tuple[str, str], sp.spmatrix]:
    """dt Phi = -curl(circulations of E), E = cE (b x U0) for an azimuthal
    background flow U0 = U(r, theta) phi_hat acting on the flux unknown
    itself (the mean-flow EMF of differential rotation, Curl(Cross(b, U0))
    in the induction equation): on the edges

        E_r (theta-faces) = cE U b_theta,   E_theta (r-faces) = -cE U b_r,   E_phi = 0,

    b = Phi / area, so the circulations are exactly the face fluxes times
    U (edge length x area cancel) and div b = 0 stays an identity.  E_theta
    is masked at a conducting wall (b_r = 0 there).  The cross-validation
    staggered code's _add_dr_emf, with cE = +1 its E = -U x b.  ``A_phys``
    blocks keyed (flux, flux)."""
    if _vec_comp(U0, "v_r", ct.RfR, ct.tcR).any() or _vec_comp(U0, "v_theta", ct.rcT, ct.TfT).any():
        raise NotImplementedError("staggered: Curl(Cross(b, U0)) on the flux unknown needs an azimuthal U0 "
                                  "(U0_r = U0_theta = 0)")
    m = ct.m
    blocks: Dict[Tuple[str, str], sp.spmatrix] = {}
    mask_rf, _, _ = ct.wall_masks()
    U_tf = _vec_comp(U0, "v_phi", ct.rcT, ct.TfT)
    U_rf = _vec_comp(U0, "v_phi", ct.RfR, ct.tcR)
    Cr_Phith = ct.Lr @ (_dd(cE * U_tf) @ ct.iAth)
    Cth_Phir = ct.Lth @ (_dd(-cE * U_rf) @ ct.iAr)
    if conducting:
        Cth_Phir = mask_rf @ Cth_Phir
    _add(blocks, "br", "br", 1j * m * Cth_Phir)
    _add(blocks, "bt", "bt", -(1j * m * Cr_Phith))
    _add(blocks, "bp", "br", -(ct.dr_rfc @ Cth_Phir))
    _add(blocks, "bp", "bt", ct.dth_tfc @ Cr_Phith)
    return blocks


def ct_resistive_blocks(ct: CTGrid, eta, conducting: bool
                        ) -> Dict[Tuple[str, str], sp.spmatrix]:
    """dt Phi = -curl(eta j), j = curl b on the edges from the physical b
    = Phi / area, the wall circulations masked for a conducting wall;
    ``A_phys`` blocks keyed (flux, flux)."""
    m = ct.m
    blocks: Dict[Tuple[str, str], sp.spmatrix] = {}
    mask_rf, mask_K, _ = ct.wall_masks()
    Lph, Lr, Lth = ct.Lph, ct.Lr, ct.Lth
    b_r, b_th, b_ph = ct.iAr, ct.iAth, ct.iAph
    Cph_Phir = Lph @ (eta * (ct.jphi_br @ b_r))
    Cph_Phith = Lph @ (eta * (ct.jphi_bth @ b_th))
    Cr_Phiph = Lr @ (eta * (ct.jr_bphi @ b_ph))
    Cr_Phith = Lr @ (eta * (ct.jr_bth @ b_th))
    Cth_Phir = Lth @ (eta * (ct.jth_br @ b_r))
    Cth_Phiph = Lth @ (eta * (ct.jth_bphi @ b_ph))
    if conducting:
        Cph_Phir, Cph_Phith = mask_K @ Cph_Phir, mask_K @ Cph_Phith
        Cth_Phir, Cth_Phiph = mask_rf @ Cth_Phir, mask_rf @ Cth_Phiph
    _add(blocks, "br", "br", -(ct.dthK_rf @ Cph_Phir) + 1j * m * Cth_Phir)
    _add(blocks, "br", "bt", -(ct.dthK_rf @ Cph_Phith))
    _add(blocks, "br", "bp", 1j * m * Cth_Phiph)
    _add(blocks, "bt", "bp", -(1j * m * Cr_Phiph))
    _add(blocks, "bt", "bt", -(1j * m * Cr_Phith) + (ct.drK_tf @ Cph_Phith))
    _add(blocks, "bt", "br", (ct.drK_tf @ Cph_Phir))
    _add(blocks, "bp", "br", -(ct.dr_rfc @ Cth_Phir))
    _add(blocks, "bp", "bp", -(ct.dr_rfc @ Cth_Phiph) + (ct.dth_tfc @ Cr_Phiph))
    _add(blocks, "bp", "bt", (ct.dth_tfc @ Cr_Phith))
    return blocks


def ct_lorentz_blocks(ct: CTGrid, B0, pre) -> Dict[Tuple[str, str], sp.spmatrix]:
    """dt v = pre (curl b) x B0 on the faces, j = curl b on the edges from
    the physical b = Phi / area; ``pre`` = (pre_vr, pre_vth, pre_vphi) the
    coefficient on the interior r-edges, interior theta-edges and the
    centres; ``A_phys`` blocks keyed (velocity, flux) on the full-face
    velocity DOFs.

    j is brought to the faces by the plain two-point averages (K_to_vr,
    K_to_vth, tf_to_c, rf_to_c) with B0 evaluated at the face: the
    constrained-transport recipe of the reference implementation."""
    blocks: Dict[Tuple[str, str], sp.spmatrix] = {}
    RintR, tcR_i, rcT_i, TintT, rcC, tcC = ct.RintR, ct.tcR_i, ct.rcT_i, ct.TintT, ct.rcC, ct.tcC
    pre_vr, pre_vth, pre_vphi = (_dd(p) for p in pre)
    S_vr, S_vt = ct.S_vr, ct.S_vt
    b_r, b_th, b_ph = ct.iAr, ct.iAth, ct.iAph
    B0t_vr, B0p_vr = _vec_comp(B0, "v_theta", RintR, tcR_i), _vec_comp(B0, "v_phi", RintR, tcR_i)
    B0r_vth, B0p_vth = _vec_comp(B0, "v_r", rcT_i, TintT), _vec_comp(B0, "v_phi", rcT_i, TintT)
    B0t_c, B0r_c = _vec_comp(B0, "v_theta", rcC, tcC), _vec_comp(B0, "v_r", rcC, tcC)
    jphi_vr_B0t = _dd(B0t_vr) @ ct.K_to_vr
    jphi_vth_B0r = _dd(B0r_vth) @ ct.K_to_vth
    jr_c_B0t = _dd(B0t_c) @ ct.tf_to_c
    jth_c_B0r = _dd(B0r_c) @ ct.rf_to_c
    jphi_vr_Phir = jphi_vr_B0t @ (ct.jphi_br @ b_r)
    jphi_vr_Phith = jphi_vr_B0t @ (ct.jphi_bth @ b_th)
    jth_vr_Phir = ct.jth_to_vr @ (ct.jth_br @ b_r)
    jth_vr_Phiph = ct.jth_to_vr @ (ct.jth_bphi @ b_ph)
    Fr_br = pre_vr @ (_dd(B0p_vr) @ jth_vr_Phir - jphi_vr_Phir)
    Fr_bt = pre_vr @ (-jphi_vr_Phith)
    Fr_bp = pre_vr @ (_dd(B0p_vr) @ jth_vr_Phiph)
    jphi_vth_Phir = jphi_vth_B0r @ (ct.jphi_br @ b_r)
    jphi_vth_Phith = jphi_vth_B0r @ (ct.jphi_bth @ b_th)
    jr_vth_Phiph = ct.jr_to_vth @ (ct.jr_bphi @ b_ph)
    jr_vth_Phith = ct.jr_to_vth @ (ct.jr_bth @ b_th)
    Ft_br = pre_vth @ jphi_vth_Phir
    Ft_bt = pre_vth @ (jphi_vth_Phith - _dd(B0p_vth) @ jr_vth_Phith)
    Ft_bp = pre_vth @ (-_dd(B0p_vth) @ jr_vth_Phiph)
    jr_c_Phiph = jr_c_B0t @ (ct.jr_bphi @ b_ph)
    jr_c_Phith = jr_c_B0t @ (ct.jr_bth @ b_th)
    jth_c_Phir = jth_c_B0r @ (ct.jth_br @ b_r)
    jth_c_Phiph = jth_c_B0r @ (ct.jth_bphi @ b_ph)
    Fp_bp = pre_vphi @ (jr_c_Phiph - jth_c_Phiph)
    Fp_bt = pre_vphi @ jr_c_Phith
    Fp_br = pre_vphi @ (-jth_c_Phir)
    _add(blocks, "vr", "br", S_vr.T @ Fr_br)
    _add(blocks, "vr", "bt", S_vr.T @ Fr_bt)
    _add(blocks, "vr", "bp", S_vr.T @ Fr_bp)
    _add(blocks, "vt", "br", S_vt.T @ Ft_br)
    _add(blocks, "vt", "bt", S_vt.T @ Ft_bt)
    _add(blocks, "vt", "bp", S_vt.T @ Ft_bp)
    _add(blocks, "vp", "br", Fp_br)
    _add(blocks, "vp", "bt", Fp_bt)
    _add(blocks, "vp", "bp", Fp_bp)
    return blocks


def ct_current_blocks(ct: CTGrid, J0, pre) -> Dict[Tuple[str, str], sp.spmatrix]:
    """dt v = pre J0 x b on the faces, b = Phi / area brought to the face
    of each component pair by the plain averages; ``pre`` as in
    ``ct_lorentz_blocks``; ``A_phys`` blocks keyed (velocity, flux)."""
    blocks: Dict[Tuple[str, str], sp.spmatrix] = {}
    RintR, tcR_i, rcT_i, TintT, rcC, tcC = ct.RintR, ct.tcR_i, ct.rcT_i, ct.TintT, ct.rcC, ct.tcC
    pre_vr, pre_vth, pre_vphi = (_dd(p) for p in pre)
    S_vr, S_vt = ct.S_vr, ct.S_vt
    bphi_c = ct.iAph
    bth_c = ct.tf_to_c @ ct.iAth
    br_c = ct.rf_to_c @ ct.iAr
    j0r_c, j0t_c = _vec_comp(J0, "v_r", rcC, tcC), _vec_comp(J0, "v_theta", rcC, tcC)
    _add(blocks, "vp", "bt", pre_vphi @ (_dd(j0r_c) @ bth_c))
    _add(blocks, "vp", "br", pre_vphi @ (-_dd(j0t_c) @ br_c))
    j0t_vr, j0p_vr = _vec_comp(J0, "v_theta", RintR, tcR_i), _vec_comp(J0, "v_phi", RintR, tcR_i)
    _add(blocks, "vr", "bp", S_vr.T @ (pre_vr @ (_dd(j0t_vr) @ ct.c_to_vr @ bphi_c)))
    if np.any(j0p_vr != 0):
        _add(blocks, "vr", "bt", S_vr.T @ (pre_vr @ (-_dd(j0p_vr) @ (ct.c_to_vr @ bth_c))))
    j0r_vth, j0p_vth = _vec_comp(J0, "v_r", rcT_i, TintT), _vec_comp(J0, "v_phi", rcT_i, TintT)
    _add(blocks, "vt", "bp", S_vt.T @ (pre_vth @ (-_dd(j0r_vth) @ ct.c_to_vth @ bphi_c)))
    if np.any(j0p_vth != 0):
        _add(blocks, "vt", "br", S_vt.T @ (pre_vth @ (_dd(j0p_vth) @ (ct.c_to_vth @ br_c))))
    return blocks


