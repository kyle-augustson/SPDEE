"""The spectral core: hand the IR to Dedalus v3.

    >>> from spdee.backends.dedalus import DedalusGrid, compile_dedalus
    >>> cp = compile_dedalus(system, DedalusGrid(N_r=24, N_theta=48, r_inner=0.71, r_outer=0.985), m=8)
    >>> res = cp.solve_evp(k=8, sigma=1j*omega_target)

Requires an importable Dedalus v3; without one ``compile`` raises an
ImportError naming the tier that provides it (the other backends are
unaffected).
"""

from spdee.backends.dedalus.compiler import DedalusGrid, DedalusCompiledProblem, compile_dedalus  # noqa: F401


def compile(system, grid=None, m=None, **kwargs):
    """Backend-registry entry point."""
    if grid is None or m is None:
        raise TypeError("the dedalus backend needs grid= and m=")
    return compile_dedalus(system, grid, m, **kwargs)


__all__ = ["DedalusGrid", "DedalusCompiledProblem", "compile_dedalus", "compile"]
