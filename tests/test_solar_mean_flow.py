"""The differential-rotation terms of ``solar_hydro_problem`` on every core.

Rigid-increment identity: a uniform profile Omega = 1 + eps entered as a
mean flow (U0 = eps r sin(theta) phi_hat, the U0.grad and (grad U0)^T terms)
is the uniform problem with Coriolis 2 (1 + eps) seen from the frame Omega0,
Doppler-shifted by m eps.  U0 has only a phi component, so the identity is
pointwise in r and theta and holds to round-off on every discretization:

    K_dr = K_frame + i m eps M      (K x = lambda M x, omega = -i lambda).

The shear and thermal-wind parts are checked by convergence against the
other cores and against the cross-validation code (outside this suite).
"""

from __future__ import annotations

import numpy as np
import pytest

from spdee.applications.solar_strings import RMAX, RMIN, solar_hydro_problem

M_ORDER = 8
EPS = 0.03


class _Rigid:
    """Omega / Omega0 = 1 + eps everywhere (nondimensional profile object)."""

    def __init__(self, eps):
        self.eps = eps

    def Omega(self, r, th):
        return (1.0 + self.eps) * np.ones(np.broadcast(r, th).shape)

    def dOdr(self, r, th):
        return np.zeros(np.broadcast(r, th).shape)

    dOdth = dOdr


class _Shear:
    """A smooth solar-like profile, for the compile and thermal-wind checks."""

    def Omega(self, r, th):
        return 1.0 + 0.02 * (r - 0.85) - 0.1 * np.cos(th) ** 2

    def dOdr(self, r, th):
        return 0.02 * np.ones(np.broadcast(r, th).shape)

    def dOdth(self, r, th):
        return 0.2 * np.cos(th) * np.sin(th) * np.ones(np.broadcast(r, th).shape)


def _grid(core):
    if core == "dedalus":
        pytest.importorskip("dedalus")
        from spdee.backends.dedalus import DedalusGrid
        # ncc_cutoff is an absolute amplitude cutoff on each NCC expansion: at the
        # production 1e-8 the small thermal-wind coefficient (~1e-5) loses modes
        # and the identities hold only to ~1e-8; the algebra is tested exactly.
        return (DedalusGrid(N_r=10, N_theta=24, r_inner=RMIN, r_outer=RMAX),
                {"backend": "dedalus", "ncc_cutoff": 1e-14})
    if core == "sh_fd":
        from spdee.backends.sh_fd import ShFdGrid
        return ShFdGrid(N_r=10, ell_max=14, r_inner=RMIN, r_outer=RMAX, fd_order=4), {"backend": "sh_fd"}
    from spdee.backends.fd2.grid import make_shell_grid
    if core == "staggered":
        return (make_shell_grid(N_r=11, N_theta=24, r_inner=RMIN, r_outer=RMAX),
                {"backend": "fd2", "discretization": "staggered"})
    return make_shell_grid(N_r=10, N_theta=24, r_inner=RMIN, r_outer=RMAX), {"backend": "fd2"}


def _pencil(pb, core):
    grid, kw = _grid(core)
    K, M = pb.compile(grid, m=M_ORDER, **kw).assemble_evp_matrices()
    return K.tocsr(), M.tocsr()


CORES = ["collocated", "staggered", "sh_fd", "dedalus"]


@pytest.mark.parametrize("core", ["collocated", "staggered", "sh_fd"])
def test_rigid_increment_identity(core):
    K_dr, M_dr = _pencil(solar_hydro_problem(rotation=_Rigid(EPS)), core)
    K_fr, M_fr = _pencil(solar_hydro_problem(frame_rate=1.0 + EPS), core)
    assert K_dr.shape == K_fr.shape
    assert abs(M_dr - M_fr).max() == 0.0
    scale = abs(K_fr).max()
    resid = abs(K_dr - K_fr - 1j * M_ORDER * EPS * M_fr).max() / scale
    wrong = abs(K_dr - K_fr + 1j * M_ORDER * EPS * M_fr).max() / scale
    assert resid < 1e-10, (core, resid)
    assert wrong > 1e3 * max(resid, 1e-16), (core, wrong)


@pytest.mark.parametrize("core", CORES)
def test_frame_rate_one_is_the_uniform_pencil(core):
    K0, M0 = _pencil(solar_hydro_problem(), core)
    K1, M1 = _pencil(solar_hydro_problem(frame_rate=1.0), core)
    assert abs(K0 - K1).max() == 0.0 and abs(M0 - M1).max() == 0.0


@pytest.mark.parametrize("core", CORES)
def test_thermal_wind_adds_only_the_entropy_coupling(core):
    """The thermal wind changes the pencil by grad(S0).v alone: the change
    is linear in tw_scale, and zero at tw_scale = 0."""
    base = _pencil(solar_hydro_problem(rotation=_Shear()), core)[0]
    tw0 = _pencil(solar_hydro_problem(rotation=_Shear(), entropy_gradient="thermal_wind", tw_scale=0.0), core)[0]
    tw1 = _pencil(solar_hydro_problem(rotation=_Shear(), entropy_gradient="thermal_wind", tw_scale=1.0), core)[0]
    tw2 = _pencil(solar_hydro_problem(rotation=_Shear(), entropy_gradient="thermal_wind", tw_scale=2.0), core)[0]
    scale = abs(base).max()
    assert abs(tw0 - base).max() <= 1e-14 * scale
    d1, d2 = tw1 - base, tw2 - base
    assert abs(d1).max() > 0
    assert abs(d2 - 2 * d1).max() <= 1e-14 * scale        # round-off of the full pencil


def test_rotation_string_other_than_uniform_is_refused():
    with pytest.raises(ValueError):
        solar_hydro_problem(rotation="solar")
    with pytest.raises(ValueError):
        solar_hydro_problem(entropy_gradient="thermal_wind")


def _nearest(K, M, omega_t, k=10):
    """The k eigenvalues omega = -i lambda of K x = lambda M x nearest omega_t."""
    import scipy.sparse as sps
    import scipy.sparse.linalg as spla
    K, M = sps.csc_matrix(K), sps.csc_matrix(M)
    lt = 1j * omega_t
    lu = spla.splu(sps.csc_matrix(K - lt * M))
    op = spla.LinearOperator(K.shape, matvec=lambda x: lu.solve(M @ x), dtype=complex)
    v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
    mu = spla.eigs(op, k=k, v0=v0, ncv=max(40, 3 * k + 1), tol=1e-14, return_eigenvectors=False)
    return -1j * (lt + 1.0 / mu)


def test_rigid_increment_identity_dedalus():
    """On the Dedalus core U0.grad(v) is an NCC product: multiplying by
    r sin(theta) raises the radial degree and the top coefficient is
    truncated, so the pencil identity holds only to the spectral tail
    (rows of the highest radial modes), exactly as in any Dedalus code that
    writes the mean flow this way.  The eigenvalues obey it to that tail:
    relative 4e-4 at Nr 8, 2e-7 at Nr 24, 5e-9 at Nr 32 for the modes
    nearest the m = 8 Rossby mode; Nr 24 must reach 1e-5."""
    pytest.importorskip("dedalus")
    from spdee.backends.dedalus import DedalusGrid
    g = DedalusGrid(N_r=24, N_theta=32, r_inner=RMIN, r_outer=RMAX)
    KA, MA = solar_hydro_problem(rotation=_Rigid(EPS)).compile(g, m=M_ORDER, backend="dedalus").assemble_evp_matrices()
    KB, MB = solar_hydro_problem(frame_rate=1.0 + EPS).compile(g, m=M_ORDER, backend="dedalus").assemble_evp_matrices()
    target = -2.0 / (M_ORDER + 1) - 0.01j
    oB = _nearest(KB, MB, target) + M_ORDER * EPS
    oA = _nearest(KA, MA, target + M_ORDER * EPS)
    oB = oB[np.argsort(abs(oB - target - M_ORDER * EPS))][:4]
    rel = max(np.min(abs(oA - w)) / abs(w) for w in oB)
    assert rel < 1e-5, rel


# ---- the magnetic problem under the mean flow ------------------------------------

MHD_CASES = [("collocated", "b"), ("staggered", "b"), ("sh_fd", "b"), ("dedalus", "b"), ("dedalus", "A")]


def _mhd(core, formulation, rotation=None, frame_rate=1.0, b0=2000.0):
    from spdee.applications.solar_eigenmodes import add_magnetic_field, toroidal_eqpeak_field
    from spdee.applications.solar_strings import ekman
    pb = solar_hydro_problem(rotation=rotation if rotation is not None else "uniform", frame_rate=frame_rate)
    hydro = pb.build()
    F = toroidal_eqpeak_field(b0)
    return add_magnetic_field(hydro, F.B0, Le=F.Le, eta=ekman(2e12), J0=F.J0, inv_rho_ratio=F.inv_rho_ratio,
                              bc="conducting", formulation=formulation,
                              solenoidal=(formulation == "b" and core != "staggered"),
                              mean_flow=pb.backgrounds.get("U0") if rotation is not None else None)


def _mhd_pencil(system, core, nr, nt):
    if core == "dedalus":
        pytest.importorskip("dedalus")
        from spdee.backends.dedalus import DedalusGrid, compile_dedalus
        return compile_dedalus(system, DedalusGrid(N_r=nr, N_theta=nt, r_inner=RMIN, r_outer=RMAX),
                               m=M_ORDER).assemble_evp_matrices()
    if core == "sh_fd":
        from spdee.backends import compile_problem
        from spdee.backends.sh_fd import ShFdGrid
        g = ShFdGrid(N_r=nr, ell_max=nt // 2, r_inner=RMIN, r_outer=RMAX, fd_order=4)
        return compile_problem(system, backend="sh_fd", grid=g, m=M_ORDER).assemble_evp_matrices()
    from spdee.backends.fd2.grid import make_shell_grid
    from spdee.backends.fd2.compiled_problem import compile_problem as cp
    if core == "staggered":
        from spdee.backends.fd2.staggered_compile import compile_fd2_staggered
        g = make_shell_grid(N_r=nr + 1, N_theta=nt, r_inner=RMIN, r_outer=RMAX)
        return compile_fd2_staggered(system, g, m=M_ORDER, order_r=2, order_theta=2).assemble_evp_matrices()
    return cp(system, make_shell_grid(N_r=nr, N_theta=nt, r_inner=RMIN, r_outer=RMAX), M_ORDER).assemble_evp_matrices()


def _mhd_identity_error(core, formulation, nr, nt):
    """max relative |omega_dr - (omega_frame + m eps)| over the 4 modes of the
    frame-rate problem nearest the m = 8 Rossby mode; the exact null space of
    the flux unknown (omega = 0: non-solenoidal flux states, which neither
    evolve nor Doppler shift) is excluded."""
    eps = -EPS            # shifts the Rossby target away from omega = 0, where the flux null space sits
    KA, MA = _mhd_pencil(_mhd(core, formulation, rotation=_Rigid(eps)), core, nr, nt)
    KB, MB = _mhd_pencil(_mhd(core, formulation, frame_rate=1.0 + eps), core, nr, nt)
    target = -2.0 / (M_ORDER + 1) - 0.01j
    oB = _nearest(KB, MB, target, k=16) + M_ORDER * eps
    oA = _nearest(KA, MA, target + M_ORDER * eps, k=24)
    oA = oA[abs(oA) > 1e-8]
    oB = oB[np.argsort(abs(oB - target - M_ORDER * eps))][:4]
    return max(np.min(abs(oA - w)) / abs(w) for w in oB)


@pytest.mark.parametrize("core,formulation", MHD_CASES)
def test_mhd_rigid_increment_identity(core, formulation):
    """With a toroidal field the rigid increment (U0, (grad U0)^T and the
    EMF of U0 on b) is still the frame change plus the Doppler shift m eps:
    curl(U0 x b) = -eps d_phi b for rigid U0 and solenoidal b, and in the A
    form -U0 x curl(A) = eps d_phi A - grad(U0.A), the gradient absorbed in
    the electric potential (U0.A = 0 on the walls).  These are identities
    of the continuous operators: the constrained-transport core keeps them
    to round-off (circulation form, div b = 0 exact), the spectral core to
    its truncation tail, and the collocated and spin-weighted cores (b with
    the solenoidal multiplier) to their truncation order, so there the test
    asks for the error to fall under refinement."""
    if core == "dedalus":
        pytest.importorskip("dedalus")
        assert _mhd_identity_error(core, formulation, 24, 32) < 1e-5
    elif core == "staggered":
        assert _mhd_identity_error(core, formulation, 16, 32) < 1e-9
    else:
        coarse = _mhd_identity_error(core, formulation, 12, 24)
        fine = _mhd_identity_error(core, formulation, 24, 48)
        # collocated: 12x24 -> 24x48 falls by the FD order; sh_fd (spectral in angle) is already at a few 1e-6
        # at 12x24 (4.2e-6 -> 2.2e-6), below which the refinement test has nothing left to see
        assert fine < 1e-3 and (fine < coarse / 3 or fine < 1e-5), (core, coarse, fine)
