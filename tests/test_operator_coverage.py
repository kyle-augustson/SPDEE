"""One manufactured boundary-value problem per (IR operator, core,
geometry, azimuthal order).

Each case states the operator once, as a string, and runs it on the
named core at two resolutions; ``tests/analytic/manufactured.py`` holds
the problem construction and the per-core drive, and
``scripts/operator_coverage.py`` turns the same runs into
``docs/OPERATOR_COVERAGE.md``.

A cell either converges at the core's stated order (spectral cores: an
error under a stated tolerance, decreasing with resolution) or the core
refuses the operator with a message ``REFUSALS`` pins. A cell that does
neither is a failure of the core, not a reason to skip: the operators
the cores get wrong are listed under "Failing cells" in
``docs/OPERATOR_COVERAGE.md`` and fail here.

Measured orders run both above and below the formal order of the
finite-difference cores at the matrix resolutions -- the centered
stencils superconverge on these smooth solutions, while a few
probe unknowns are still pre-asymptotic -- so the assertion is
one-sided at ``CoreSpec.min_order``, and the measured value is
reported either way.
"""

import numpy as np
import pytest
import sympy as sp

from analytic import vector_calculus as vc
from analytic.manufactured import (
    CASES, CASES_BY_NAME, CORE_SPECS, GEOMETRIES, R_INNER,
    _err, _fd_grid, _stag_solve, _stag_state, _stag_values,
    build_problem, measure, run_once,
)
from spdee.ir import r, theta
from spdee.problem import DEFAULT_NAMESPACE

M_VALUES = (0, 1, 2)

# Operators a core declines, and the message its own compiler must
# raise. A key is (operator, core), or (operator, core, geometry) when
# only one geometry declines. Removing an entry turns the cell back into
# a convergence check, so a core that gains an operator fails here until
# its entry goes.
REFUSALS = {
    # v_r of a regular vector is r^(l-1) Y_l at the origin, not a regular
    # scalar of the ball basis; the projection is a wall row only.
    ("Dot(r_hat, vector)", "dedalus", "ball"): "unit-vector projections are available on the ball",
}


def _pattern(operator, core, geometry):
    return REFUSALS.get((operator, core, geometry), REFUSALS.get((operator, core)))


def _cells():
    for case in CASES:
        for core in CORE_SPECS:
            for geometry in GEOMETRIES:
                for m in M_VALUES:
                    yield pytest.param(case.name, core, geometry, m,
                                       id=f"{case.name}-{core}-{geometry}-m{m}")


# The full matrix is 624 boundary-value solves at two resolutions each,
# about an hour; it is marked slow so the default suite stays fast. Run
# it with -m slow, or regenerate the matrix with
# scripts/operator_coverage.py.
@pytest.mark.slow
@pytest.mark.parametrize("operator,core,geometry,m", list(_cells()))
def test_operator_coverage(operator, core, geometry, m):
    case = CASES_BY_NAME[operator]
    pattern = _pattern(operator, core, geometry)
    if pattern is not None:
        with pytest.raises((NotImplementedError, TypeError), match=pattern):
            run_once(case, core, geometry, m)
        return
    outcome = measure(case, core, geometry, m)
    spec = CORE_SPECS[core]
    expect = ("below {:g}".format(spec.tolerance) if spec.order is None
              else f"order >= {spec.min_order}")
    assert outcome.status == "PASS", (
        f"{operator} on {core} {geometry} m={m}: {outcome.status} "
        f"({outcome.reason}); errors {outcome.errors} at N={outcome.resolutions}, "
        f"measured order {outcome.order:.2f}, expected {expect}")


def _solve_staggered(u_sym, m, n):
    """Solve Lap(u) = Lap(u_sym) on the C-grid and return the max error."""
    case = CASES_BY_NAME["Lap(vector)"]
    pb, _f_u, _f_w = build_problem(case, "shell", m, "evolution")
    cm = pb.compile(_fd_grid("shell", n, 2), m=m, backend="fd2",
                    discretization="staggered", order_r=2, order_theta=2)
    x, live = _stag_solve(cm, _stag_state(cm, {"u": vc.lap(u_sym, m)}, "shell", m))
    err, scale = 0.0, 0.0
    for field in cm.layout.fields:
        sl = slice(field.offset, field.offset + field.size)
        exact = _stag_values(u_sym, field, cm.grids, "shell", m)
        scale = max(scale, float(np.abs(exact).max()))
        err = max(err, _err(x[sl][live[sl]], exact[live[sl]]))
    return err / scale


def test_staggered_vector_laplacian_is_the_general_operator():
    """The C-grid Lap(vector) converges on a field with Div(u) != 0.

    The strain Gram is Lap + (1/3) Grad Div, so it stands in for Lap only
    on the divergence-free manifold: built for every Lap(u), it converged
    at order 2 on a solenoidal manufactured field and not at all on one
    built from the same generator with Div(u) != 0, which is the carrier
    every staggered cell with a vector unknown used to fail through. An
    unknown that carries no divergence constraint now gets the general
    operator and both fields converge at the scheme's order 2.
    """
    m = 1
    wall = (r - R_INNER) * (1 - r)
    base = (r * sp.sin(theta)) ** m * sp.cos(r * sp.cos(theta))
    solenoidal = vc.curl(vc.Vec(r * wall * base, 0, 0), m)
    general = vc.grad(base, m) * wall
    assert sp.simplify(vc.div(solenoidal, m)) == 0
    assert sp.simplify(vc.div(general, m)) != 0
    fine, coarse = 24, 12
    for name, field in (("solenoidal", solenoidal), ("general", general)):
        e = [_solve_staggered(field, m, n) for n in (coarse, fine)]
        order = float(np.log(e[0] / e[1]) / np.log(fine / coarse))
        assert order > 1.5, f"{name} field: order {order:.2f}, errors {e}"


def test_trace_is_not_an_ir_operator():
    """Trace has no IR node, so no equation string can reach the C-grid
    trace map; the trace piece of a viscous stress is written as
    ``bg_vec * Div(v)`` or ``Grad(Div(v))``, both covered above."""
    assert "Trace" not in DEFAULT_NAMESPACE and "trace" not in DEFAULT_NAMESPACE
    import spdee.ir.operators as ops
    assert not hasattr(ops, "Trace")
