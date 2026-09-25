"""spdee.backends.fd2.sweep — assemble matrices over a set of (m, params) cases.

spdee builds matrices and leaves the parallelism to the caller. The
unit of work for a parameter study is: assemble the EVP matrices for
one azimuthal mode ``m`` and one set of physical parameter values.
Any outer driver (``multiprocessing``, ``mpi4py``, a job array, a
notebook loop) can map over it; nothing here imports an executor.

``assemble_case(system, grid, m, params, ...)``
    One ``(m, params)`` cell → a ``CompiledProblem``.

``iter_assembled(system, grid, cases, ...)``
    Serial generator over many cells, yielding ``(key, CompiledProblem)``.

Symbolic parameters
-------------------
Write the system with free SymPy symbols for the swept parameters, e.g.

    Ek = sympy.Symbol("Ek")
    momentum = Equation(Dt(v) - Ek * Lap(v) + Grad(p), name="mom")

and assemble concrete values with ``params={Ek: 1e-4}``.
``spdee.ir.tasks.bind_parameters`` performs the substitution (scalar
coefficients and SymPy metadata; numeric / tabular background
components are evaluated independently).

Driver example (the parallelism lives in the caller)::

    from multiprocessing import Pool
    cases = [((m, ek), m, {Ek: ek}) for m in (1, 2, 3)
             for ek in (1e-4, 1e-3, 1e-2)]
    def work(case):
        key, m, params = case
        cp = assemble_case(system, grid, m, params)
        K, M = cp.assemble_evp_matrices()
        return key, K, M
    with Pool() as pool:
        results = dict(pool.map(work, cases))
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional, Tuple

from spdee.ir.equations import EquationSystem
from spdee.ir.tasks import bind_parameters


def assemble_case(
    system: EquationSystem,
    grid,
    m: int,
    params: Optional[Dict] = None,
    *,
    order_r: int = 4,
    order_theta: int = 4,
    prefer_aux_free: bool = True,
    discretization=None,
):
    """Assemble one ``(m, params)`` cell into a CompiledProblem.

    Thin wrapper over ``compile_problem`` that names the sweep use case.
    """
    from spdee.backends.fd2.compiled_problem import compile_problem
    return compile_problem(
        system, grid, m,
        order_r=order_r, order_theta=order_theta,
        prefer_aux_free=prefer_aux_free,
        discretization=discretization,
        params=params,
    )


def iter_assembled(
    system: EquationSystem,
    grid,
    cases: Iterable[Tuple[object, int, Optional[Dict]]],
    *,
    order_r: int = 4,
    order_theta: int = 4,
    prefer_aux_free: bool = True,
) -> Iterator[Tuple[object, "object"]]:
    """Assemble ``(key, m, params)`` cases, yielding ``(key, CompiledProblem)``.

    The symbolic walk (augmentation, equation tasks, BC tasks) is cached
    per distinct ``m``; parameter binding and matrix assembly run per
    case. The walk is cheap next to the per-case operator assembly, so
    this is a convenience for iterating a sweep rather than a speedup.
    Uses the collocated discretization.
    """
    from spdee.backends.fd2.compiler import compile_fd2
    from spdee.ir.walker import walk_equation_system, walk_boundary_conditions
    from spdee.ir.augment import augment_equation_system
    from spdee.backends.fd2.compiled_problem import CompiledProblem

    walk_cache: Dict[int, Tuple] = {}
    for key, m, params in cases:
        if m not in walk_cache:
            aug = augment_equation_system(
                system, prefer_aux_free=prefer_aux_free,
            )
            tasks = walk_equation_system(aug, m)
            bc_tasks = walk_boundary_conditions(aug, m)
            walk_cache[m] = (aug, tasks, bc_tasks)
        aug, tasks, bc_tasks = walk_cache[m]
        bound = bind_parameters(tasks, params) if params else tasks
        matrices = compile_fd2(
            aug, grid, m,
            order_r=order_r, order_theta=order_theta,
            pre_walked_tasks=bound,
        )
        yield key, CompiledProblem(matrices=matrices, bc_tasks=bc_tasks)
