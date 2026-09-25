"""Spin-weighted spherical harmonics in (theta, phi) at fixed m, finite
differences in r; shell or ball (``r_inner=0``, ``docs/ball_sh_fd.md``).

    >>> from spdee.backends.sh_fd import ShFdGrid, compile_sh_fd
    >>> cp = compile_sh_fd(system, ShFdGrid(N_r=48, ell_max=32, r_inner=0.71, r_outer=0.985), m=8)
    >>> res = cp.solve_evp(k=8, sigma=1j*omega_target)
"""

from spdee.backends.sh_fd.angular import AngularVasil, sYlm_theta, kcoef  # noqa: F401
from spdee.backends.sh_fd.compiler import ShFdGrid, ShFdCompiledProblem, compile_sh_fd  # noqa: F401


def compile(system, grid=None, m=None, **kwargs):
    """Backend-registry entry point."""
    if grid is None or m is None:
        raise TypeError("the sh_fd backend needs grid= (ShFdGrid) and m=")
    return compile_sh_fd(system, grid, m, **kwargs)


__all__ = ["AngularVasil", "sYlm_theta", "kcoef", "ShFdGrid", "ShFdCompiledProblem",
           "compile_sh_fd", "compile"]
