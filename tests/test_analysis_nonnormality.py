"""Non-normality of the compiled pencils (spdee.analysis.nonnormality): the
reduced operator of a normal pencil has zero Henrici departure and equal
numerical and spectral abscissae; the collocated inertial ball is non-normal
in the volume norm (its Grad/Div pair is not adjoint) while the staggered
one, whose pencil is exactly W-skew, has a numerical abscissa equal to its
spectral abscissa to rounding.
"""

import numpy as np

from spdee import Problem, analysis as an
from spdee.backends.fd2.grid import make_ball_grid


def _inertial(discretization, N_r, N_theta):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    grid = make_ball_grid(N_r=N_r, N_theta=N_theta)
    if discretization == "staggered":
        return pb.compile(grid, m=2, discretization="staggered", order_r=2, order_theta=2)
    return pb.compile(grid, m=2, order_r=4, order_theta=4)


def test_staggered_inertial_pencil_is_skew_in_volume_norm():
    """The C-grid Coriolis and Grad/Div blocks are W-skew, so the evolution
    operator on the constraint manifold is W-normal: every finite eigenvalue
    is neutral, the numerical abscissa vanishes with the spectral one, and
    Henrici's departure is at rounding."""
    cp = _inertial("staggered", 9, 12)
    K, M = cp.assemble_evp_matrices()
    W = an.natural_weight(cp, M)
    red = an.reduced_operator(K, M, sigma=0.3 + 0.2j, W=W)
    h = an.henrici(red.A)
    na = an.numerical_abscissa(red.A)
    print(f"BENCH | staggered ball 8x12 m=2: {red.k} finite modes, Henrici relative "
          f"{h['relative']:.1e}, numerical abscissa {na['numerical']:.1e}, spectral "
          f"{na['spectral']:.1e}")
    assert np.max(np.abs(red.lam.real)) < 1e-8
    assert h["relative"] < 1e-6
    assert abs(na["numerical"]) < 1e-6 and abs(na["excess"]) < 1e-6
    c = an.mode_nonnormality(K, M, red.lam[:5], W=W, reduced=red)
    assert np.all(c < 1.0 + 1e-4)


def test_collocated_inertial_pencil_is_nonnormal():
    """The collocated pair is not adjoint: damped and growing spurious modes,
    a numerical abscissa above the spectral one, and per-mode excess
    resolvent well above one on the ill-conditioned modes."""
    cp = _inertial("collocated", 12, 18)
    K, M = cp.assemble_evp_matrices()
    W = an.natural_weight(cp, M)
    red = an.reduced_operator(K, M, sigma=0.3 + 0.2j, W=W)
    h = an.henrici(red.A)
    na = an.numerical_abscissa(red.A)
    fov = an.numerical_range(red.A, 32)
    c = an.mode_nonnormality(K, M, red.lam[:20], W=W, reduced=red)
    print(f"BENCH | collocated ball 12x18 m=2: {red.k} finite modes, Henrici relative "
          f"{h['relative']:.2f}, numerical abscissa {na['numerical']:.3f} against spectral "
          f"{na['spectral']:.3f}, field of values radius {np.max(np.abs(fov)):.2f}, per-mode "
          f"excess resolvent max {c.max():.1f}")
    assert h["relative"] > 0.1
    assert na["excess"] > 0.05
    assert c.max() > 2.0
