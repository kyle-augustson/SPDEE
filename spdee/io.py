"""The spectrum file contract: one npz per solve.

    omega : complex128 (n,)   omega/Omega0; retrograde Re < 0, damped Im < 0,
                              growing Im > 0 (omega = -i lambda)
    lam   : complex128 (n,)   pencil eigenvalue, K x = lam M x
    resid : float64 (n,)      ||K x - lam M x|| / (||K x|| + |lam| ||M x||)
    V     : complex128 (N, n) raw pencil vectors, columns aligned with omega
    meta  : JSON string       core, resolution, physics, conventions, layout,
                              spdee commit and version, shift

File name: spdee_<core>_<tag>_m<m>_nr<N_r>_l<ell>_nu<nu>.npz
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp

__all__ = ["pencil_residuals", "spectrum_meta", "save_spectrum", "load_spectrum",
           "spectrum_filename"]


def pencil_residuals(K, M, lam, V) -> np.ndarray:
    K = sp.csr_matrix(K)
    KV = K @ V
    MV = sp.csr_matrix(M) @ V if M is not None else V
    num = np.linalg.norm(KV - MV * lam[None, :], axis=0)
    den = np.linalg.norm(KV, axis=0) + np.abs(lam) * np.linalg.norm(MV, axis=0)
    den = np.where(den == 0, 1.0, den)
    return num / den


def _git_commit(path: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short=12", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def spectrum_meta(cp, **extra) -> Dict[str, Any]:
    """Metadata for a compiled problem of any core, plus caller extras."""
    import spdee
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta: Dict[str, Any] = dict(
        core=getattr(cp, "discretization", "unknown"), m=int(cp.m),
        omega_convention="omega = -i*lambda; omega/Omega0 units; retrograde Re<0; damped Im<0; growing Im>0",
        Omega0_nHz=456.0, spdee_version=getattr(spdee, "__version__", None),
        spdee_commit=_git_commit(here),
    )
    core = meta["core"]
    if core == "sh_fd":
        g = cp.grid
        meta.update(N_r=int(cp.Nr), ell_max=int(g.ell_max), fd_order=int(g.fd_order),
                    r_inner=float(g.r_inner), r_outer=float(g.r_outer),
                    r_nodes="uniform" if g.nodes is None else "custom",
                    layout=dict(components=["-", "+", "r"],
                                spin="v_- = (v_theta - i v_phi)/sqrt2, v_+ = (v_theta + i v_phi)/sqrt2, v_r",
                                l=f"{cp.m}..{g.ell_max}", index="a_l*N_r + i (l-major, r inner)",
                                blocks=[(name, kind, off, size) for name, kind, off, size in cp.layout]))
    elif core == "dedalus":
        g = cp.grid
        meta.update(N_r=int(g.N_r), N_theta=int(g.N_theta), geometry=g.geometry,
                    r_inner=float(g.r_inner), r_outer=float(g.r_outer),
                    layout=dict(pencil="Dedalus L_min/M_min (K = L_min, M = M_min, lambda = -s)",
                                unknowns=list(cp.unknown_names),
                                taus=[t.name for t in cp.tau_fields]))
    elif core == "staggered":
        meta.update(layout=dict(fields=[(f.name, f.location, f.offset, f.size) for f in cp.layout.fields]))
    meta.update(extra)
    return meta


def save_spectrum(path: str, cp, lam, V, meta_extra: Optional[Dict[str, Any]] = None,
                  resid: Optional[np.ndarray] = None) -> str:
    lam = np.asarray(lam, dtype=np.complex128)
    V = np.asarray(V, dtype=np.complex128)
    if V.ndim == 1:
        V = V[:, None]
    if resid is None:
        K, M = cp.assemble_evp_matrices()
        resid = pencil_residuals(K, M, lam, V)
    meta = spectrum_meta(cp, **(meta_extra or {}))
    np.savez(path, omega=-1j * lam, lam=lam, resid=np.asarray(resid, dtype=float), V=V,
             meta=json.dumps(meta, default=str))
    return path


def load_spectrum(path: str):
    z = np.load(path, allow_pickle=False)
    return dict(omega=z["omega"], lam=z["lam"], resid=z["resid"], V=z["V"],
                meta=json.loads(str(z["meta"])))


def _nu_tag(nu: float) -> str:
    mant, exp = f"{nu:.6e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


def spectrum_filename(core: str, tag: str, m: int, N_r: int, ell: int, nu: float) -> str:
    return f"spdee_{core}_{tag}_m{m}_nr{N_r}_l{ell}_nu{_nu_tag(nu)}.npz"
