"""Tests for spdee.backends dispatch."""

import pytest

from spdee.backends import available_backends, compile_problem
from spdee.ir import Unknown, Equation, BoundaryCondition, EquationSystem, Lap
from spdee.ir.equations import Boundary


def _trivial_system():
    p = Unknown.Scalar("p")
    eq = Equation(Lap(p), name="poisson")
    bc = BoundaryCondition(Boundary.R_OUTER, p, name="dirichlet")
    return EquationSystem([eq], [bc])


def test_available_backends():
    names = available_backends()
    assert "fd2" in names
    assert "sh_fd" in names
    assert "dedalus" in names


def test_compile_unknown_backend_raises():
    sys_ = _trivial_system()
    with pytest.raises(ValueError, match="Unknown backend"):
        compile_problem(sys_, backend="not_a_backend")


@pytest.mark.parametrize("backend", ["sh_fd", "dedalus"])
def test_grid_backends_require_grid(backend):
    sys_ = _trivial_system()
    with pytest.raises(TypeError):
        compile_problem(sys_, backend=backend)
