"""Tests for spdee.backends.fd2.grid."""

import numpy as np
import pytest

from spdee.backends.fd2.grid import (
    Grid, uniform_r, uniform_theta,
    make_ball_grid, make_shell_grid,
)


# ===========================================================================
# uniform_r / uniform_theta
# ===========================================================================

def test_uniform_r_basic():
    r = uniform_r(10, 0.0, 1.0)
    assert r.shape == (10,)
    assert r[0] == 0.0
    assert r[-1] == 1.0
    assert np.allclose(np.diff(r), 1.0 / 9.0)


def test_uniform_r_requires_inner_lt_outer():
    with pytest.raises(ValueError, match="r_inner < r_outer"):
        uniform_r(10, 1.0, 0.5)


def test_uniform_theta_excludes_poles():
    theta = uniform_theta(20)
    assert theta.shape == (20,)
    assert theta[0] > 0
    assert theta[-1] < np.pi
    # Cell-centered: midpoints of N_theta equal cells
    expected = (np.arange(20) + 0.5) * np.pi / 20
    assert np.allclose(theta, expected)


def test_uniform_theta_min_N():
    with pytest.raises(ValueError, match="N_theta must be >= 2"):
        uniform_theta(1)


# ===========================================================================
# Grid construction
# ===========================================================================

def test_make_ball_grid():
    g = make_ball_grid(N_r=8, N_theta=6, r_outer=2.0)
    assert g.is_ball
    assert not g.is_shell
    assert g.N_r == 8
    assert g.N_theta == 6
    assert g.r[0] == 0.0
    assert g.r[-1] == 2.0
    assert g.r_inner == 0.0
    assert g.r_outer == 2.0
    assert g.shape == (8, 6)
    assert g.size == 48


def test_make_shell_grid():
    g = make_shell_grid(N_r=10, N_theta=8, r_inner=0.3, r_outer=1.0)
    assert g.is_shell
    assert not g.is_ball
    assert g.r[0] == pytest.approx(0.3)
    assert g.r[-1] == pytest.approx(1.0)


def test_grid_mesh():
    g = make_ball_grid(N_r=5, N_theta=4)
    R, Theta = g.mesh()
    assert R.shape == (5, 4)
    assert Theta.shape == (5, 4)
    # First row should be all r=0
    assert np.all(R[0, :] == 0.0)
    # First column should be all theta[0]
    assert np.allclose(Theta[:, 0], g.theta[0])


# ===========================================================================
# Validation
# ===========================================================================

def test_grid_invalid_geometry():
    with pytest.raises(ValueError, match="geometry must be"):
        Grid(r=np.linspace(0, 1, 10), theta=uniform_theta(5), geometry="cube")


def test_grid_ball_requires_r0_at_origin():
    with pytest.raises(ValueError, match="ball geometry requires r"):
        Grid(r=np.linspace(0.1, 1, 10), theta=uniform_theta(5), geometry="ball")


def test_grid_shell_rejects_origin():
    with pytest.raises(ValueError, match="shell geometry requires r"):
        Grid(r=np.linspace(0, 1, 10), theta=uniform_theta(5), geometry="shell")


def test_grid_r_must_be_monotone():
    bad_r = np.array([0.0, 0.5, 0.3, 1.0])
    with pytest.raises(ValueError, match="monotonically increasing"):
        Grid(r=bad_r, theta=uniform_theta(5), geometry="ball")


def test_grid_theta_excludes_poles():
    bad_theta = np.array([0.0, 0.5, 1.0, 2.0])  # includes theta=0
    with pytest.raises(ValueError, match="theta=0"):
        Grid(r=uniform_r(5, 0, 1), theta=bad_theta, geometry="ball")


def test_grid_theta_excludes_south_pole():
    bad_theta = np.array([0.1, 0.5, 1.0, np.pi])  # includes theta=pi
    with pytest.raises(ValueError, match="theta=pi"):
        Grid(r=uniform_r(5, 0, 1), theta=bad_theta, geometry="ball")


def test_grid_repr():
    g = make_ball_grid(N_r=8, N_theta=6)
    s = repr(g)
    assert "ball" in s
    assert "N_r=8" in s
    assert "N_theta=6" in s
