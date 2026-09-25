"""End-to-end smoke test of the public Stage-1 API.

Exercises everything a user would touch from `import spdee` and confirms
the IR composes the way the README example claims.
"""

import sympy as sp

import spdee


def test_public_api_quickstart():
    """Run the README quickstart pattern, end-to-end."""
    # Fields
    rho0 = spdee.Background.Scalar("rho0", expr=sp.exp(-spdee.r))
    B0 = spdee.Background.Vector(
        "B0", v_phi=spdee.r * sp.sin(spdee.theta)
    )
    v = spdee.Unknown.Vector("v")
    p = spdee.Unknown.Scalar("p")

    # Equations (linear)
    momentum = spdee.Equation(
        rho0 * spdee.Dt(v) + spdee.Grad(p) - spdee.Cross(B0, spdee.Curl(v)),
        name="momentum",
    )
    div_free = spdee.Equation(spdee.Div(v), name="div_free")

    # BCs
    bc_no_penetration = spdee.BoundaryCondition(
        spdee.Boundary.R_OUTER, spdee.Dot(spdee.r_hat, v), name="no-penetration"
    )

    # System
    sys_ = spdee.EquationSystem([momentum, div_free], [bc_no_penetration])
    assert sys_.unknown_names == frozenset({"v", "p"})
    assert "rho0" in sys_.background_names
    assert "B0" in sys_.background_names
    assert len(sys_.equations) == 2
    assert len(sys_.boundary_conditions) == 1
