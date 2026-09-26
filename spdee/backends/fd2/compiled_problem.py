"""spdee.backends.fd2.compiled_problem — assembled problem with BVP/EVP solvers.

A CompiledProblem holds the CompiledMatrices from compile_fd2 (A_total,
B_total, b_total, layout, grid) and the BCTasks from the IR's BC walker.
It installs ball-mode origin regularity (the reduction of the i_r=0
unknowns to the regular degrees of freedom, see ball_origin.py),
boundary conditions and gauge constraints by row replacement, then
solves either the BVP ``A x = b`` (solve_bvp) or the generalized EVP
``K x = λ M x`` (solve_evp, shift-invert through spdee.solvers).

Supported BCs, all at radial boundaries: scalar Dirichlet,
vector-projection Dirichlet, scalar-linear-from-tensor (e.g.
stress-free), radial Neumann, the projected curl / cross combination
(a tangential electric field) and their linear combinations (Robin,
Navier slip, the normal stress; see ir/tasks.py, BCTask). A row is the
per-node sum of the coefficient at the wall times the row each term
alone would install, on the host the record names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spdee.backends.fd2.ball_origin import (
    ell0_projection_weights, origin_anchor_sigma, origin_shape, tensor_origin_basis,
)
from spdee.backends.fd2.compiler import CompiledMatrices
from spdee.backends.fd2.stencils import _is_uniform
from spdee.ir.tasks import (
    BCTask, BC_DIRICHLET_SCALAR, BC_DIRICHLET_VECTOR_PROJECTION,
    BC_SCALAR_LINEAR_FROM_TENSOR, BC_NEUMANN_SCALAR_RADIAL,
    BC_PROJECTION_CURL_CROSS, BC_LINEAR_COMBINATION, BC_TERM_KINDS,
    BC_UNIT_R_HAT, BC_UNIT_THETA_HAT, BC_UNIT_PHI_HAT,
    BC_LOC_R_INNER, BC_LOC_R_OUTER,
)

# Intel MKL PARDISO (pip install pypardiso) is used for the BVP direct
# solve only when SPDEE_USE_PARDISO=1, otherwise scipy's SuperLU.  Importing
# pypardiso loads libmkl_rt, whose FFTW wrappers shadow the FFTW that Dedalus
# links, after which Dedalus cannot create transform plans in this process.
_HAVE_PARDISO = False
if os.environ.get("SPDEE_USE_PARDISO", "0") == "1":
    try:
        from pypardiso import spsolve as _pardiso_spsolve
        _HAVE_PARDISO = True
    except ImportError:
        pass


def _sparse_solve(A, b):
    """Solve A x = b with pypardiso when enabled, else spla.spsolve.

    pypardiso accepts float64 only. Matrices are stored as complex128,
    so when A and b are effectively real (max |imag| < 1e-14) they are
    cast down for the solve and the result cast back to complex;
    genuinely complex systems (e.g. m ≠ 0) go to SuperLU.
    """
    if not _HAVE_PARDISO:
        return spla.spsolve(A, b)
    if A.dtype == np.float64 and b.dtype == np.float64:
        return _pardiso_spsolve(A, b)
    if np.iscomplexobj(A.data):
        A_imag_max = float(np.max(np.abs(A.data.imag))) if A.nnz else 0.0
    else:
        A_imag_max = 0.0
    if np.iscomplexobj(b):
        b_imag_max = float(np.max(np.abs(b.imag))) if b.size else 0.0
    else:
        b_imag_max = 0.0
    if A_imag_max < 1e-14 and b_imag_max < 1e-14:
        A_real = A.astype(np.float64) if A.dtype != np.float64 else A
        b_real = b.real.astype(np.float64) if np.iscomplexobj(b) else b.astype(np.float64)
        x_real = _pardiso_spsolve(A_real, b_real)
        if np.iscomplexobj(b):
            return x_real.astype(np.complex128)
        return x_real
    return spla.spsolve(A, b)



@dataclass
class EvpResult:
    """Result of `CompiledProblem.solve_evp`.

    Attributes
    ----------
    eigenvalues : np.ndarray (complex, length ≤ k)
        Raw generalized eigenvalues λ of ``K x = λ M x``.
    eigenvectors : list of dict, optional
        If requested, a list of {unknown_name: (N_r, N_theta) array}
        dicts, one per eigenvalue. None otherwise.
    """
    eigenvalues: np.ndarray
    eigenvectors: Optional[List[Dict[str, np.ndarray]]]

    @staticmethod
    def eigenvalues_to_omega(lam) -> np.ndarray:
        r"""Map raw generalized eigenvalues λ to physical frequencies ω.

        This is the single definition of the λ→ω convention; consumers
        that interpret a `solve_evp` result as a time-evolution
        frequency go through it. A wrong sign flips the damping axis
        while leaving |Re(ω)| intact, which frequency-only checks
        cannot see.

        Derivation. The compiler routes ``Dt(unknown)`` terms into
        ``M = B_total`` with their literal equation sign and every other
        term into ``K = A_total``, and the solver returns λ of

            K x = λ M x.

        For the viscous-decay equation as written in an `Equation`,

            Dt(v) - Ek * Lap(v) = 0,

        ``M = +I`` and ``K = -Ek·Lap``, so ``λ = -Ek·eig(Lap)``. The
        discrete Lap is negative-definite (``eig(Lap) ≈ -k²``), giving
        ``λ = +Ek·k² > 0``. Physically ``∂_t v = +Ek·∇²v`` decays: with
        ``v ~ e^{-iωt}`` the rate is ``-iω = -Ek·k²``, i.e.
        ``Im(ω) = -Ek·k² < 0``. Matching,

            -iω = -Ek·k²  and  λ = +Ek·k²   ⟹   ω = -i·λ,

        so the convention is λ = +iω ⟺ ω = -iλ:

            Re(ω) = +Im(λ)          (oscillation frequency)
            Im(ω) = -Re(λ)          (growth rate; < 0 ⇒ damped)

        A neutral spectrum (e.g. inviscid Rossby modes) contains both
        ``λ = ±i|ω|`` and cannot distinguish the sign; a damped mode
        does, and selects ω = -iλ (a retrograde damped mode lies on the
        ``Im(λ) < 0`` branch with ``Re(ω) < 0`` and ``Im(ω) < 0``).

        This map applies to the time-evolution interpretation only.
        Spatial eigenproblems written without ``Dt`` (Helmholtz-style
        ``Lap(v) = λ v``) put Lap directly in K and return the spatial
        eigenvalue ``λ = -κ²``; there is no ω.

        Parameters
        ----------
        lam : array_like of complex
            Raw eigenvalues from `EvpResult.eigenvalues` or
            `solve_evp_from_matrices`.

        Returns
        -------
        np.ndarray of complex
            Physical frequencies ω = -iλ.
        """
        return -1j * np.asarray(lam)

    def omega(self) -> np.ndarray:
        """Physical frequencies ω = -iλ of this result's eigenvalues
        (see `eigenvalues_to_omega`)."""
        return EvpResult.eigenvalues_to_omega(self.eigenvalues)


@dataclass
class CompiledProblem:
    """A compiled problem ready to be solved.

    Attributes
    ----------
    matrices : CompiledMatrices
        Assembled matrices, layout and grid.
    bc_tasks : list of BCTask
        Boundary conditions from the IR's BC walker.
    impenetrability_host : str
        Which wall row Dot(r̂, v) = 0 replaces on a vector v that has a
        divergence constraint (see `_bc_host`): 'constraint' hosts it
        on the constraint's wall row and keeps the wall-normal momentum
        row as the pressure's natural condition; 'momentum' hosts it on
        the σ = 0 wall row of v's own equation and keeps the constraint
        at the wall; 'auto' (default) is 'constraint' at m = 0 and
        'momentum' at m ≠ 0.
    """
    matrices: CompiledMatrices
    bc_tasks: List[BCTask]
    impenetrability_host: str = "auto"

    def solve_bvp(self) -> Dict[str, np.ndarray]:
        """Solve ``A_total x = b_total`` with origin regularity (ball
        mode), BCs and gauge constraints installed by row replacement.

        Returns
        -------
        dict {unknown_name: (N_r, N_theta) array}
            The primary unknowns unpacked from the flat state.
        """
        from spdee.backends.fd2.profiling import perf_now, record, is_enabled
        prof = is_enabled()
        cm = self.matrices
        if prof: t0 = perf_now()
        A = sp.lil_matrix(cm.A_total, dtype=complex)
        b = np.array(cm.b_total, dtype=complex, copy=True)
        if prof: record("solve_bvp/lil_copy", perf_now() - t0)

        # Origin rows first, then BCs (so BCs at r_outer are free to
        # replace their rows), then gauge rows.
        if cm.grid.is_ball:
            if prof: t0 = perf_now()
            self._apply_ball_origin_regularity(A, b)
            if prof: record("solve_bvp/origin_regularity", perf_now() - t0)
        if prof: t0 = perf_now()
        self._check_bc_host_collisions()
        for bct in self.bc_tasks:
            self._apply_bc_row_replacement(bct, A, b)
        if prof: record("solve_bvp/bcs", perf_now() - t0)
        if cm.gauge_unknowns:
            if prof: t0 = perf_now()
            self._apply_gauge_constraints(A, b)
            if prof: record("solve_bvp/gauge", perf_now() - t0)

        if prof: t0 = perf_now()
        A_csr = A.tocsr()
        if prof: record("solve_bvp/lil_to_csr", perf_now() - t0)
        if prof: t0 = perf_now()
        x = _sparse_solve(A_csr, b)
        if prof: record("solve_bvp/sparse_solve", perf_now() - t0)
        if prof: t0 = perf_now()
        out = cm.layout.unpack(x, role="primary")
        if prof: record("solve_bvp/unpack", perf_now() - t0)
        return out

    def assemble_evp_matrices(
        self,
        mass_equations=None,
    ) -> "Tuple[sp.csr_matrix, sp.csr_matrix]":
        """Assemble the (K, M) pair of the generalized EVP.

        This is the σ-independent part of `solve_evp`: K is A_total
        with origin regularity, BC rows and gauge rows installed; M
        comes from B_total (Path A) or the per-equation masses (Path B).
        The origin anchor rows carry the same ℓ=0 projection in K and
        M (so Dt and Path-B masses keep their weight at r=0); every
        other replaced row is zeroed in M. A frequency scan assembles
        once and factors ``K − σM`` per target.

        Returns
        -------
        (K_csr, M_csr) : tuple of scipy.sparse.csr_matrix, both (N, N)
        """
        from spdee.backends.fd2.profiling import perf_now, record, is_enabled
        prof = is_enabled()
        cm = self.matrices
        if prof: t0 = perf_now()
        K = sp.lil_matrix(cm.A_total, dtype=complex)
        b_dummy = np.zeros(K.shape[0], dtype=complex)
        M_lil = self._mass_lil(K.shape[0], mass_equations)
        overwritten: set = set()
        if cm.grid.is_ball:
            self._apply_ball_origin_regularity(K, b_dummy, overwritten, M=M_lil)
        self._check_bc_host_collisions()
        for bct in self.bc_tasks:
            self._apply_bc_row_replacement(bct, K, b_dummy, overwritten)
        if cm.gauge_unknowns:
            self._apply_gauge_constraints(K, b_dummy, overwritten)
        for row in overwritten:
            M_lil.rows[row] = []
            M_lil.data[row] = []
        K_csr = K.tocsr()
        M_csr = M_lil.tocsr()
        if prof: record("solve_evp/build_KM", perf_now() - t0)
        return K_csr, M_csr

    def _mass_lil(self, N: int, mass_equations) -> "sp.lil_matrix":
        """Build M as a LIL matrix (Path A from B_total, Path B from the
        per-equation masses; see `solve_evp`)."""
        from spdee.backends.fd2.profiling import perf_now, record, is_enabled
        prof = is_enabled()
        cm = self.matrices
        mass_terms = self._resolve_mass_terms(mass_equations)
        if prof: t0 = perf_now()
        if cm.B_total.nnz > 0:
            M_lil = cm.B_total.tolil().copy()
        else:
            M_lil = sp.lil_matrix((N, N), dtype=complex)
            for eq_name, uk_name in mass_terms.items():
                row_slice = cm.equation_row_slices[eq_name]
                uk_entry = cm.layout.entry(uk_name)
                uk_offset = uk_entry.offset
                eq_mass = cm.equation_mass_matrices.get(eq_name, None)
                eq_size = row_slice.stop - row_slice.start
                uk_size = uk_entry.size
                if eq_size != uk_size:
                    raise ValueError(
                        f"For EVP, equation {eq_name!r} (size {eq_size}) "
                        f"and unknown {uk_name!r} (size {uk_size}) must have "
                        f"the same size — λ acting on a vector unknown "
                        f"requires a vector equation."
                    )
                if eq_mass is None:
                    for i in range(eq_size):
                        M_lil[row_slice.start + i, uk_offset + i] = 1.0
                else:
                    em = eq_mass.tocoo()
                    base_row = row_slice.start
                    for r0, c0, v in zip(em.row, em.col, em.data):
                        M_lil[base_row + int(r0), uk_offset + int(c0)] = v
        if prof: record("solve_evp/build_M", perf_now() - t0)
        return M_lil

    def _resolve_mass_terms(self, mass_equations):
        """Resolve `mass_equations` into a {equation: unknown} dict for
        Path B; return None in Path A (B_total non-empty)."""
        cm = self.matrices
        eq_names = list(cm.equation_row_slices.keys())
        uk_names = list(cm.layout.primary_names)
        if cm.B_total.nnz > 0:
            return None
        if mass_equations is None:
            if len(eq_names) == 1 and len(uk_names) == 1:
                return {eq_names[0]: uk_names[0]}
            raise ValueError(
                f"solve_evp: with {len(eq_names)} equations and "
                f"{len(uk_names)} unknowns, the eigenvalue's target "
                f"unknown(s) must be specified via mass_equations, or "
                f"write Dt(...) in an equation. "
                f"Equations: {eq_names}. Unknowns: {uk_names}."
            )
        if isinstance(mass_equations, dict):
            mass_terms = dict(mass_equations)
        else:
            mass_terms = {en: en for en in mass_equations}
        for eq_name, uk_name in mass_terms.items():
            if eq_name not in cm.equation_row_slices:
                raise ValueError(
                    f"mass_equations references unknown equation "
                    f"{eq_name!r}; available: "
                    f"{sorted(cm.equation_row_slices.keys())}"
                )
            try:
                cm.layout.entry(uk_name)
            except KeyError:
                raise ValueError(
                    f"mass_equations[{eq_name!r}]={uk_name!r} is not a "
                    f"known unknown."
                )
        return mass_terms

    @staticmethod
    def solve_evp_from_matrices(
        K_csr: "sp.csr_matrix",
        M_csr: "sp.csr_matrix",
        sigma: complex,
        k: int,
        return_eigenvectors: bool = False,
        method: str = "arpack",
        factor: str = "scipy",
        **solver_options,
    ) -> np.ndarray:
        """Shift-invert eigensolve of a pre-assembled (K, M) at one shift.

        (K, M) come from `assemble_evp_matrices`; this forms ``K − σM``,
        factors it and finds `k` eigenvalues near σ. ``method`` and
        ``factor`` select the eigensolver and inner factorization in
        :mod:`spdee.solvers` (``'arpack'``/``'scipy'`` by default;
        ``'bbjd'``, ``'fplus'``, ``'cudss'``, ``'cupy'`` are the
        alternatives); extra keyword options are forwarded to
        :func:`spdee.solvers.shift_invert`.

        Returns the eigenvalue array (length ≤ k, infinite eigenvalues
        mapped to NaN), or ``(evals, evecs)`` with `return_eigenvectors`.
        """
        from spdee.solvers import shift_invert
        from spdee.backends.fd2.profiling import perf_now, record, is_enabled
        prof = is_enabled()
        if prof: t0 = perf_now()
        try:
            evals, evecs = shift_invert(
                K_csr, M_csr, sigma, k, method=method, factor=factor,
                return_vectors=return_eigenvectors, **solver_options,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"solve_evp_from_matrices: failed at σ={sigma} "
                f"(method={method!r}, factor={factor!r}). Underlying error: {e}"
            ) from e
        if prof: record(f"solve_evp/{method}", perf_now() - t0)
        evals = np.where(np.isfinite(evals), evals, np.complex128(np.nan))
        if return_eigenvectors:
            return evals, evecs
        return evals

    def solve_evp(
        self,
        k: int = 6,
        sigma: Optional[complex] = None,
        return_eigenvectors: bool = False,
        mass_equations: Optional[List[str]] = None,
        method: str = "arpack",
        factor: str = "scipy",
        **solver_options,
    ) -> "EvpResult":
        """Solve the generalized eigenvalue problem ``K x = λ M x``.

        K is `solve_bvp`'s assembled operator: A_total with origin
        regularity, BCs and gauge rows installed. M is built one of two
        ways, and every row K replaced is zeroed in M:

        Path A: ``Dt(unknown)`` appears in some equation (optionally
        with a scalar coefficient, ``rho0 * Dt(v)``). The compiler
        accumulates the corresponding mass operator into B_total, which
        is used as M. This is the only path that honours
        variable-coefficient masses. Equations without Dt get zero
        rows in M automatically.

        Path B: B_total is empty. `mass_equations` names which
        equations host the eigenvalue side and which unknown they
        multiply; M is the per-equation natural mass (``diag(r²)`` for
        Lap-based equations, ``diag(r)`` for Grad/Div-based, identity
        otherwise) with identity coupling on the unknown's column
        block. No coefficient is applied.

        For a Helmholtz problem ``Lap u + κ² u = 0`` (Path B), K is
        ``r²·Lap`` and M is ``r²·I``, so the returned eigenvalues are
        ``λ = -κ²`` and ``κ_n = √(-λ_n)``.

        For a time-evolution problem (Dt present), convert the raw λ
        with `EvpResult.eigenvalues_to_omega` (ω = -iλ): Re(ω) is the
        oscillation frequency and Im(ω) the growth rate (< 0 ⇒ damped).

        Parameters
        ----------
        k : int
            Number of eigenvalues to compute.
        sigma : complex, optional
            Shift-invert target. Zero-mass rows (BCs, constraints) make
            M singular; infinite eigenvalues map to μ ≈ 0 in the
            shifted-inverse spectrum and are returned as NaN. If None,
            ``scipy.sparse.linalg.eigs(K, k, M=M)`` is called directly;
            a non-default `method` or `factor` or any solver option is
            rejected on that path.
        return_eigenvectors : bool
            If True, also return the unpacked eigenvector dictionaries.
        method, factor : str
            Eigensolver and inner factorization, see :mod:`spdee.solvers`
            (defaults ``'arpack'`` and ``'scipy'``). Further keyword
            options are forwarded to :func:`spdee.solvers.shift_invert`.
        mass_equations : dict[str, str], list[str], or None
            Path B only (ignored in Path A):
              - dict {eq_name: unknown_name}: explicit mapping; equations
                not listed get zero rows in M.
              - list[str]: equation names, each sharing its name with a
                primary unknown of the same kind.
              - None: allowed only with exactly one equation and one
                primary unknown.

        Returns
        -------
        EvpResult
            `.eigenvalues` (complex, length ≤ k, NaN for infinite
            eigenvalues) and, if requested, `.eigenvectors`.

        Notes
        -----
        Coupled velocity-pressure systems with a divergence constraint
        also produce a few N-dependent spurious modes from the
        collocated-FD inf-sup deficit; physical modes converge with N
        while the spurious ones drift. Use
        `spdee.backends.fd2.evp_filter.filter_spurious_eigenvalues`
        (twin-grid comparison) to separate them.
        """
        K_csr, M_csr = self.assemble_evp_matrices(mass_equations=mass_equations)
        cm = self.matrices
        if sigma is None:
            if method != "arpack" or factor != "scipy" or solver_options:
                raise ValueError(
                    f"solve_evp: sigma=None calls scipy.sparse.linalg.eigs "
                    f"directly; method={method!r}, factor={factor!r} and "
                    f"solver options {sorted(solver_options)} apply to the "
                    f"shift-invert path only. Pass a shift sigma."
                )
            eigs_out = spla.eigs(
                K_csr, k=k, M=M_csr,
                return_eigenvectors=return_eigenvectors,
            )
            if return_eigenvectors:
                evals, evecs = eigs_out
                vectors = [cm.layout.unpack(evecs[:, i], role="primary")
                           for i in range(evecs.shape[1])]
                return EvpResult(eigenvalues=evals, eigenvectors=vectors)
            return EvpResult(eigenvalues=eigs_out, eigenvectors=None)
        out = CompiledProblem.solve_evp_from_matrices(
            K_csr, M_csr, sigma, k=k,
            return_eigenvectors=return_eigenvectors,
            method=method, factor=factor, **solver_options,
        )
        if return_eigenvectors:
            evals, evecs = out
            vectors = [cm.layout.unpack(evecs[:, i], role="primary")
                       for i in range(evecs.shape[1])]
            return EvpResult(eigenvalues=evals, eigenvectors=vectors)
        return EvpResult(eigenvalues=out, eigenvectors=None)

    def _origin_row_base(self, name: str) -> int:
        """First row of the unknown's origin rows: the row block of its
        paired equation, whose i_r=0 rows hold the equation at r=0 (the
        unknown's column offset when no pairing is recorded)."""
        cm = self.matrices
        eq = cm.paired_equation_of_unknown.get(name)
        if eq is None:
            return cm.layout.entry(name).offset
        return cm.equation_row_slices[eq].start

    def _hosts_on_constraint(self, unknown: str, location: str) -> bool:
        """Whether Dot(r̂, unknown) = 0 at `location` is hosted on the
        wall row of the unknown's divergence constraint.

        At a wall node of a v-p system there are four equations, one
        impenetrability condition and four unknowns, so one equation
        row gives way. 'constraint' drops the constraint (Div v = 0)
        there and keeps the wall-normal momentum row: on an impenetrable
        wall the normal momentum equation is the pressure's boundary
        condition, ∂_n p = n̂·(forces − Dt v), and its one-sided ∂_r p
        sees the r-sawtooth that centred stencils cannot, so the
        pressure checkerboard of the m = 0 collocated system is
        removed. 'momentum' drops the normal momentum row instead and
        keeps the constraint at the wall, the textbook closure, which
        on the first-order (v_r, p) system of an inviscid problem is
        the more accurate one by 10-100x at fixed resolution (Zhang
        targets, docs/ball_collocated.md); at m ≠ 0 the azimuthal
        gradient i m p / (r sin θ) in the tangential momentum rows
        already fixes the wall pressure pointwise, so 'auto' takes
        'constraint' only at m = 0, where nothing else does. A scalar
        BC of the constraint's own unknown (a Dirichlet pressure) at
        the same wall claims that row, and the projection then falls
        back to the momentum row.
        """
        cm = self.matrices
        mode = self.impenetrability_host
        if mode == "auto":
            mode = "constraint" if cm.m == 0 else "momentum"
        if mode not in ("constraint", "momentum"):
            raise ValueError(
                f"impenetrability_host must be 'auto', 'constraint' or "
                f"'momentum'; got {mode!r}"
            )
        constraint = cm.divergence_constraint_of_vector.get(unknown)
        if mode != "constraint" or constraint is None:
            return False
        scalar = next(
            (u for u, e in cm.paired_equation_of_unknown.items() if e == constraint),
            None,
        )
        for other in self.bc_tasks:
            if (other.kind in (BC_DIRICHLET_SCALAR, BC_NEUMANN_SCALAR_RADIAL,
                               BC_LINEAR_COMBINATION)
                    and other.unknown_name == scalar
                    and other.location == location):
                return False
        return True

    def _bc_host(self, bct: BCTask):
        """(row_base, key) of the rows a BC replaces: `row_base` is the
        first flat row of the hosting spin block and `key` =
        (equation, σ or None, location) identifies it for collisions.

        Every row belongs to an equation, so a BC replaces rows of the
        equation paired with its unknown, whatever the column layout: a
        scalar BC the wall row of the scalar's equation; a vector
        projection the wall row of the spin block that carries its
        component (θ̂ → σ = +1, φ̂ → σ = −1, r̂ → σ = 0); a
        tensor-derived scalar BC its metadata host. Impenetrability on
        a vector with a divergence constraint is hosted on the
        constraint's wall row when `_hosts_on_constraint` says so.
        """
        cm = self.matrices
        N = cm.grid.N_r * cm.grid.N_theta
        spin_offset = {-1: 0, 0: N, +1: 2 * N}
        if bct.kind in (BC_SCALAR_LINEAR_FROM_TENSOR, BC_LINEAR_COMBINATION):
            unknown = bct.metadata["host_unknown"]
            sigma = bct.metadata["host_sigma"]
        elif bct.kind == BC_DIRICHLET_VECTOR_PROJECTION:
            unknown = bct.unknown_name
            sigma = _UNIT_SIGMA[bct.unit]
            if sigma == 0 and self._hosts_on_constraint(unknown, bct.location):
                constraint = cm.divergence_constraint_of_vector[unknown]
                start = cm.equation_row_slices[constraint].start
                return start, (constraint, None, bct.location)
        elif bct.kind == BC_PROJECTION_CURL_CROSS:
            unknown = bct.unknown_name
            sigma = _UNIT_SIGMA[bct.unit]
        else:
            unknown = bct.unknown_name
            sigma = None
        eq = cm.paired_equation_of_unknown.get(unknown)
        base = self._origin_row_base(unknown)
        if sigma is None:
            return base, (eq or unknown, None, bct.location)
        return base + spin_offset[sigma], (eq or unknown, sigma, bct.location)

    def _gauge_host(self, name: str):
        """(row, key) of the gauge row of scalar unknown `name`: the
        (i_r, j = 0) row of its paired equation at the first interior
        radial node inside the outer wall (N_r − 2 on a ball, whose
        i_r = 0 rows are the origin rows; 1 on a shell), so it never
        meets a wall or origin row of the equation."""
        cm = self.matrices
        i_gauge = cm.grid.N_r - 2 if cm.grid.is_ball else 1
        eq = cm.paired_equation_of_unknown.get(name)
        return self._origin_row_base(name) + i_gauge, (eq or name, None, i_gauge)

    @staticmethod
    def _set_row(mat: sp.lil_matrix, row: int, entries: Dict[int, complex]) -> None:
        """Replace one LIL row by the (column → value) entries."""
        cols = sorted(c for c, v in entries.items() if v != 0)
        mat.rows[row] = cols
        mat.data[row] = [entries[c] for c in cols]

    def _apply_ball_origin_regularity(
        self, A: sp.lil_matrix, b: np.ndarray,
        overwritten_rows: Optional[set] = None,
        M: Optional[sp.lil_matrix] = None,
    ) -> None:
        """Reduce the i_r=0 unknowns to the regular degrees of freedom
        (ball_origin.py) by replacing the i_r=0 rows of each unknown's
        paired equation.

        The compiled i_r=0 rows hold the equation at r=0 with every
        operator replaced by its r→0 limit. For an unknown with one
        origin DOF (scalar m=0, vector m=0 or |m|=1, shape functions
        s_σ(θ) from `origin_shape`) the rows become
          - one anchor at (σ_a, i_r=0, j=0), σ_a from
            `origin_anchor_sigma`: the ℓ=0 projection of the Cartesian
            projection of the equation,
                Σ_j w_j Σ_σ s_σ(θ_j) · row(σ, 0, j),
            applied to A, to M (when given) and to b, so coefficients,
            signs, pointwise, identity, Dt and source terms enter as in
            the equation;
          - ties on every other (σ, 0, j):
                v^σ(0, θ_j) − (s_σ(θ_j)/s_σa(θ_0)) · v^σa(0, θ_0) = 0,
            with zero rows in M.
        An unknown that vanishes at the origin (scalar m≠0, vector
        |m|≥2) receives Dirichlet-zero rows, zero in M. The tie and
        Dirichlet rows are added to `overwritten_rows` if given; the
        anchor keeps its mass and is not. A tensor unknown, with several
        origin DOFs, goes through `_apply_tensor_origin_regularity`.
        """
        cm = self.matrices
        grid = cm.grid
        m = cm.m
        N_r, N_theta = grid.N_r, grid.N_theta
        N = N_r * N_theta
        w_proj = ell0_projection_weights(grid.theta)
        mats = [A] if M is None else [A, M]
        for name in cm.layout.primary_names:
            entry = cm.layout.entry(name)
            kind = entry.kind
            if kind == "tensor":
                self._apply_tensor_origin_regularity(
                    name, A, b, overwritten_rows, M, w_proj)
                continue
            shape = origin_shape(m, kind, grid.theta)
            sigmas = [0] if kind == "scalar" else [-1, 0, +1]
            row_base = self._origin_row_base(name)
            col_base = entry.offset

            def row_of(s_idx, j, row_base=row_base):
                return row_base + s_idx * N + j * N_r

            def col_of(s_idx, j, col_base=col_base):
                return col_base + s_idx * N + j * N_r

            if shape is None:
                for s_idx in range(len(sigmas)):
                    for j in range(N_theta):
                        row = row_of(s_idx, j)
                        self._set_row(A, row, {col_of(s_idx, j): 1.0})
                        b[row] = 0.0
                        if M is not None:
                            self._set_row(M, row, {})
                        if overwritten_rows is not None:
                            overwritten_rows.add(row)
                continue

            # Projections of the original rows, taken before any row of
            # this unknown is overwritten.
            proj = [{} for _ in mats]
            b_proj = 0j
            for s_idx, sigma in enumerate(sigmas):
                for j in range(N_theta):
                    w = w_proj[j] * shape[sigma][j]
                    row = row_of(s_idx, j)
                    b_proj += w * b[row]
                    for acc, mat in zip(proj, mats):
                        for c, v in zip(mat.rows[row], mat.data[row]):
                            acc[c] = acc.get(c, 0.0) + w * v
            anchor_sigma = origin_anchor_sigma(m, kind)
            a_idx = sigmas.index(anchor_sigma)
            anchor_row = row_of(a_idx, 0)
            anchor_col = col_of(a_idx, 0)
            s_anchor = float(shape[anchor_sigma][0])
            for acc, mat in zip(proj, mats):
                self._set_row(mat, anchor_row, acc)
            b[anchor_row] = b_proj
            for s_idx, sigma in enumerate(sigmas):
                for j in range(N_theta):
                    row = row_of(s_idx, j)
                    if row == anchor_row:
                        continue
                    ratio = float(shape[sigma][j]) / s_anchor
                    self._set_row(A, row, {col_of(s_idx, j): 1.0, anchor_col: -ratio})
                    b[row] = 0.0
                    if M is not None:
                        self._set_row(M, row, {})
                    if overwritten_rows is not None:
                        overwritten_rows.add(row)

    def _apply_tensor_origin_regularity(
        self, name: str, A: sp.lil_matrix, b: np.ndarray,
        overwritten_rows: Optional[set], M: Optional[sp.lil_matrix],
        w_proj: np.ndarray,
    ) -> None:
        """The origin rows of a tensor unknown with d origin DOFs
        (`tensor_origin_basis`, d = 3, 2, 1 at |m| = 0, 1, 2): the nine
        i_r = 0 values T_{σ_a σ_b}(0, θ_j) lie in the span of the basis
        s^(k), so d anchor rows carry the projections
        Σ_j w_j Σ_ab s^(k)_ab(θ_j) row(ab, 0, j) of the equation (in A, M
        and b) at d well-conditioned positions (pivoted QR of the basis),
        and every other origin position q is tied to the anchors,
        T(q) − Σ_k C[k, q] T(p_k) = 0 with C = S_anchor^{-1} S, zero in M.
        A tensor that vanishes at the origin gets Dirichlet-zero rows."""
        from scipy.linalg import qr
        from spdee.backends.fd2.layout import SPIN_COMPONENTS_TENSOR
        cm = self.matrices
        grid = cm.grid
        N_r, N_theta = grid.N_r, grid.N_theta
        N = N_r * N_theta
        basis = tensor_origin_basis(cm.m, grid.theta)
        row_base = self._origin_row_base(name)
        col_base = cm.layout.entry(name).offset
        positions = [(c, j) for c in range(len(SPIN_COMPONENTS_TENSOR)) for j in range(N_theta)]

        def row_of(c, j):
            return row_base + c * N + j * N_r

        def col_of(c, j):
            return col_base + c * N + j * N_r

        if not basis:
            for c, j in positions:
                row = row_of(c, j)
                self._set_row(A, row, {col_of(c, j): 1.0})
                b[row] = 0.0
                if M is not None:
                    self._set_row(M, row, {})
                if overwritten_rows is not None:
                    overwritten_rows.add(row)
            return
        mats = [A] if M is None else [A, M]
        S = np.array([[shape[key][j] for key in SPIN_COMPONENTS_TENSOR for j in range(N_theta)]
                      for shape in basis])
        d = S.shape[0]
        _q, _r, piv = qr(S, pivoting=True)
        anchors = [positions[k] for k in piv[:d]]
        C = np.linalg.solve(S[:, piv[:d]], S)
        projections = []
        for k in range(d):
            proj = [{} for _ in mats]
            b_proj = 0j
            for c, key in enumerate(SPIN_COMPONENTS_TENSOR):
                for j in range(N_theta):
                    w = w_proj[j] * basis[k][key][j]
                    row = row_of(c, j)
                    b_proj += w * b[row]
                    for acc, mat in zip(proj, mats):
                        for col, v in zip(mat.rows[row], mat.data[row]):
                            acc[col] = acc.get(col, 0.0) + w * v
            projections.append((proj, b_proj))
        for k, (c, j) in enumerate(anchors):
            row = row_of(c, j)
            for acc, mat in zip(projections[k][0], mats):
                self._set_row(mat, row, acc)
            b[row] = projections[k][1]
        for q, (c, j) in enumerate(positions):
            if (c, j) in anchors:
                continue
            row = row_of(c, j)
            entries = {col_of(c, j): 1.0}
            for k, (ca, ja) in enumerate(anchors):
                entries[col_of(ca, ja)] = entries.get(col_of(ca, ja), 0.0) - C[k, q]
            self._set_row(A, row, entries)
            b[row] = 0.0
            if M is not None:
                self._set_row(M, row, {})
            if overwritten_rows is not None:
                overwritten_rows.add(row)

    def _check_bc_host_collisions(self) -> None:
        """Raise if two BCs would replace the same boundary host row.

        Each BC replaces one row per θ_j of the host block `_bc_host`
        resolves, keyed (equation, spin block, location); a second BC
        on the same host row would silently overwrite the first.
        Unknown BC kinds are left to the dispatcher.
        """
        host_keys = {}
        for bct in self.bc_tasks:
            _, key = self._bc_host(bct)
            if key in host_keys:
                raise ValueError(
                    f"BC host-row collision: both BCs {host_keys[key]!r} "
                    f"and {bct.name!r} target the same boundary row "
                    f"(equation={key[0]!r}, spin_sigma={key[1]!r}, "
                    f"location={key[2]!r}). Each BC installs by replacing "
                    f"a single row, so two BCs cannot share a host row "
                    f"without silently overwriting one of them. Restructure "
                    f"the BC set so that each member targets a distinct "
                    f"(equation, spin block, location) triple."
                )
            host_keys[key] = bct.name

    def _apply_gauge_constraints(
        self,
        A: sp.lil_matrix,
        b: np.ndarray,
        overwritten_rows: Optional[set] = None,
    ) -> None:
        """Install the discrete volume-integral constraint ∫ p dV = 0 for
        each scalar gauge unknown at m = 0.

        The constraint fixes the constant mode, which exists only at
        m = 0 (a constant is not an e^{imφ} field, and the discrete Grad
        has no null vector at m ≠ 0). At m ≠ 0 nothing is installed: the
        row Σ w p_m = 0 would replace an equation by a constraint the
        solution does not satisfy and corrupt v and p at O(1).

        dV = 2π r² sin θ dr dθ (the 2π drops out). Quadrature: composite
        trapezoid on the radial nodes, midpoint rule in θ with the cell
        edges at the midpoints between nodes and at the poles
        (`_gauge_quadrature_weights`; on the uniform grids these are h,
        h/2 at the ends, and Δθ). The r=0 weight vanishes, so the row has
        no entry at i_r=0.

        The row replaced is the one `_gauge_host` names: the (i_r, j=0)
        row of the gauge unknown's paired equation at the first interior
        node inside the outer wall, which no BC or origin row can claim.
        The dropped equation (one continuity row of a v-p system) is
        the one the divergence theorem makes redundant. Row indices
        written are added to `overwritten_rows` if given.
        """
        cm = self.matrices
        if cm.m != 0:
            return
        grid = cm.grid
        N_r = grid.N_r
        N_theta = grid.N_theta
        w_r, w_theta = _gauge_quadrature_weights(grid.r, grid.theta)
        w_2d = (
            w_r[:, None] * w_theta[None, :]
            * (grid.r[:, None] ** 2) * np.sin(grid.theta)[None, :]
        )
        for gname in cm.gauge_unknowns:
            entry = cm.layout.entry(gname)
            if entry.kind != "scalar":
                raise NotImplementedError(
                    f"Gauge constraint only supports scalar unknowns; "
                    f"{gname!r} is {entry.kind}"
                )
            field_offset = entry.offset
            target_row, _ = self._gauge_host(gname)
            A.rows[target_row] = []
            A.data[target_row] = []
            if overwritten_rows is not None:
                overwritten_rows.add(target_row)
            cols = []
            vals = []
            for i in range(N_r):
                for j in range(N_theta):
                    col = field_offset + j * N_r + i
                    w = w_2d[i, j]
                    if w != 0.0:
                        cols.append(col)
                        vals.append(complex(w))
            for c, v in zip(cols, vals):
                A[target_row, c] = v
            b[target_row] = 0.0

    def _apply_bc_row_replacement(
        self,
        bct: BCTask,
        A: sp.lil_matrix,
        b: np.ndarray,
        overwritten_rows: Optional[set] = None,
    ) -> None:
        """Apply one BC by row replacement: the row of a single-shape kind
        (`_term_rows`), or the per-node sum of the wall coefficients times
        the term rows of a BC_LINEAR_COMBINATION, on the host row
        `_bc_host` names. Row indices written are added to
        `overwritten_rows` if given."""
        i_r = self._resolve_boundary_radial_index(bct)
        if bct.kind == BC_LINEAR_COMBINATION:
            rows = self._combination_rows(bct, i_r)
        elif bct.kind in BC_TERM_KINDS:
            rows = self._term_rows(bct, i_r)
        else:
            raise NotImplementedError(f"BC kind {bct.kind!r} not supported")
        cm = self.matrices
        N_r = cm.grid.N_r
        row_base, _ = self._bc_host(bct)
        value_arr = self._evaluate_bc_value(bct, i_r)
        for j, entries in enumerate(rows):
            row_idx = row_base + j * N_r + i_r
            self._set_row(A, row_idx, entries)
            b[row_idx] = value_arr[j]
            if overwritten_rows is not None:
                overwritten_rows.add(row_idx)

    def _resolve_boundary_radial_index(self, bct: BCTask) -> int:
        """Radial index of a BC location (0 at r_inner, N_r−1 at r_outer);
        user BCs at r=0 in ball geometry are rejected."""
        cm = self.matrices
        grid = cm.grid
        if grid.geometry == "ball" and bct.location == BC_LOC_R_INNER:
            raise ValueError(
                f"BC {bct.name!r}: cannot impose a user BC at r=0 in ball "
                f"geometry (regularity is handled internally)."
            )
        if bct.location == BC_LOC_R_INNER:
            return 0
        if bct.location == BC_LOC_R_OUTER:
            return grid.N_r - 1
        raise NotImplementedError(
            f"BC location {bct.location!r} not supported"
        )

    def _evaluate_at_wall(self, component, i_r: int) -> np.ndarray:
        """A background component on the θ grid at radial index `i_r`,
        shape (N_theta,) complex."""
        grid = self.matrices.grid
        theta_arr = grid.theta
        r_arr = np.full_like(theta_arr, grid.r[i_r])
        value_2d = component.evaluate_on_grid(r_arr.reshape(1, -1), theta_arr.reshape(1, -1))
        return np.asarray(value_2d).ravel().astype(complex)

    def _evaluate_bc_value(self, bct: BCTask, i_r: int) -> np.ndarray:
        """The BC value on the θ grid at radial index `i_r`, shape
        (N_theta,) complex; zeros when the BC is homogeneous."""
        if bct.value is None:
            return np.zeros_like(self.matrices.grid.theta, dtype=complex)
        return self._evaluate_at_wall(bct.value, i_r)

    def _wall_stencil(self, i_r: int):
        """(radial indices, ∂_r weights) of the one-sided derivative over
        ``order_r + 1`` nodes at the wall node `i_r`, evaluated at the
        actual node positions (valid on non-uniform r grids)."""
        from spdee.backends.fd2.stencils import _lagrange_derivative_weights
        cm = self.matrices
        N_r = cm.grid.N_r
        stencil_size = cm.bundle.order_r + 1
        if i_r == 0:
            stencil_ir = list(range(stencil_size))
        elif i_r == N_r - 1:
            stencil_ir = list(range(N_r - stencil_size, N_r))
        else:
            raise NotImplementedError(
                f"BC at interior radial index {i_r} not supported "
                f"(only i_r=0 or i_r=N_r-1)"
            )
        r_arr = cm.grid.r
        stencil_r = np.array([float(r_arr[k]) for k in stencil_ir])
        w_d1 = _lagrange_derivative_weights(stencil_r, float(r_arr[i_r]), order=1)
        return stencil_ir, w_d1

    def _term_rows(self, term: BCTask, i_r: int) -> List[Dict[int, complex]]:
        """The homogeneous row of one single-shape term at every wall node:
        a list over θ_j of {column: coefficient} on the unknown's columns.

        Scalar Dirichlet: ``x[j, i_r] = value``. Radial Neumann
        ``(∂_r s)(r_wall, θ_j)``: the one-sided ∂_r weights of
        `_wall_stencil` on the unknown's radial slice. A vector
        projection, in the spin basis v^± = (v_θ ± i v_φ)/√2:
        r̂·v = v^0, θ̂·v = (v^+ + v^-)/√2, φ̂·v = i(v^- − v^+)/√2, so over
        the columns [σ=−1, 0, +1] the patterns are r̂: [0, 1, 0],
        θ̂: [1/√2, 0, 1/√2], φ̂: [+i/√2, 0, −i/√2]. A tensor projection
        (stress-free) and the projected curl / cross combination (a
        tangential electric field) are the discretisation of a scalar
        SymPy expression in the placeholders of the unknown's spin
        components (`_rows_from_scalar_expr`); the Cross part of the
        latter is pointwise in the background's spin components at the
        wall nodes.
        """
        cm = self.matrices
        N_r, N_theta = cm.grid.N_r, cm.grid.N_theta
        block = N_r * N_theta
        spin_block = {-1: 0, 0: block, +1: 2 * block}
        if term.kind == BC_DIRICHLET_SCALAR:
            offset = cm.layout.field_slice(term.unknown_name).start
            return [{offset + j * N_r + i_r: 1.0} for j in range(N_theta)]
        if term.kind == BC_NEUMANN_SCALAR_RADIAL:
            offset = cm.layout.field_slice(term.unknown_name).start
            stencil_ir, w_d1 = self._wall_stencil(i_r)
            return [{offset + j * N_r + k: complex(w) for k, w in zip(stencil_ir, w_d1)}
                    for j in range(N_theta)]
        if term.kind == BC_DIRICHLET_VECTOR_PROJECTION:
            offset = cm.layout.field_slice(term.unknown_name).start
            if term.unit not in _UNIT_SPIN_COEFFICIENTS:
                raise NotImplementedError(
                    f"BC {term.name!r}: unsupported unit vector {term.unit!r}"
                )
            coefs = _UNIT_SPIN_COEFFICIENTS[term.unit]
            return [{offset + spin_block[sigma] + j * N_r + i_r: c
                     for sigma, c in coefs.items() if c != 0}
                    for j in range(N_theta)]
        if term.kind == BC_SCALAR_LINEAR_FROM_TENSOR:
            placeholder_to_id = {}
            for uname, spin_map in term.metadata["placeholders"].items():
                for sigma, fn_obj in spin_map.items():
                    placeholder_to_id[fn_obj.func] = (uname, sigma)
            return self._rows_from_scalar_expr(
                term.metadata["scalar_expr"], placeholder_to_id, i_r)
        if term.kind == BC_PROJECTION_CURL_CROSS:
            return self._curl_cross_rows(term, i_r)
        raise NotImplementedError(f"BC kind {term.kind!r} not supported")

    def _rows_from_scalar_expr(
        self, scalar_expr, placeholder_to_id: Dict, i_r: int,
    ) -> List[Dict[int, complex]]:
        """Rows of a scalar SymPy expression, linear in the placeholders
        ``v_sigma(r, θ)`` of one or more vector unknowns' spin components
        and their first r / θ derivatives, at the wall node `i_r`. The
        expression is split into terms ``coef(r, θ) · action(placeholder)``
        (`_decompose_bc_scalar_expr`); for each θ_j the coefficient is
        evaluated at (r_wall, θ_j) and lands on the placeholder's
        spin-block columns: at (i_r, j) for a pointwise term, over the
        one-sided radial stencil for ∂_r, and over the j-th row of the
        dense D1_θ for ∂_θ."""
        import sympy as _sp
        from spdee.ir.fields import r as r_sym, theta as t_sym
        cm = self.matrices
        grid = cm.grid
        N_r, N_theta = grid.N_r, grid.N_theta
        block = N_r * N_theta
        spin_block = {-1: 0, 0: block, +1: 2 * block}
        r_boundary = float(grid.r[i_r])
        terms = _decompose_bc_scalar_expr(scalar_expr, placeholder_to_id)
        stencil_ir, w_d1 = self._wall_stencil(i_r)
        D1_theta = cm.bundle.d1_theta_dense
        rows = []
        for j in range(N_theta):
            theta_j = float(grid.theta[j])
            col_coefs: Dict[int, complex] = {}
            for term in terms:
                ang_coef_expr = term["angular_coef"]
                ang_coef_val = complex(
                    _sp.N(ang_coef_expr.subs([(r_sym, r_boundary), (t_sym, theta_j)]))
                )
                if ang_coef_val == 0:
                    continue
                ph_offset = cm.layout.field_slice(term["unknown_name"]).start
                sigma = term["sigma"]
                action = term["action"]
                if action == "pointwise":
                    col = ph_offset + spin_block[sigma] + j * N_r + i_r
                    col_coefs[col] = col_coefs.get(col, 0.0) + ang_coef_val
                elif action == "d_r":
                    for k, w in enumerate(w_d1):
                        col = ph_offset + spin_block[sigma] + j * N_r + stencil_ir[k]
                        col_coefs[col] = (
                            col_coefs.get(col, 0.0) + ang_coef_val * complex(w)
                        )
                elif action == "d_theta":
                    row_weights = D1_theta[j, :]
                    for j_other in range(N_theta):
                        w = row_weights[j_other]
                        if w == 0:
                            continue
                        col = ph_offset + spin_block[sigma] + j_other * N_r + i_r
                        col_coefs[col] = (
                            col_coefs.get(col, 0.0) + ang_coef_val * complex(w)
                        )
                else:
                    raise NotImplementedError(
                        f"BC term action {action!r} not supported"
                    )
            rows.append(col_coefs)
        return rows

    def _curl_cross_rows(self, term: BCTask, i_r: int) -> List[Dict[int, complex]]:
        """Rows of ``unit · X`` at the wall, X the constant-coefficient
        combination of Curl(u) and Cross(u, B0) / Cross(B0, u) terms of
        BC_PROJECTION_CURL_CROSS. The Curl terms are projected from the
        spin-basis curl formulas (`_curl_spin_formulas`, divided by r) and
        discretised as a scalar expression; the Cross terms are pointwise,
        Cross(B0, u)^σ from the background's spin components at the wall
        nodes, negated for Cross(u, B0)."""
        import sympy as _sp
        from spdee.backends.common.field_eval import evaluate_vector_background_spin_2d
        from spdee.backends.fd2.curl_cross_sparse import _curl_spin_formulas
        from spdee.ir.fields import r as r_sym, theta as t_sym
        cm = self.matrices
        grid = cm.grid
        N_r, N_theta = grid.N_r, grid.N_theta
        block = N_r * N_theta
        spin_block = {-1: 0, 0: block, +1: 2 * block}
        unit_coefs = _UNIT_SPIN_COEFFICIENTS[term.unit]
        scalar_expr = _sp.S.Zero
        placeholder_to_id = {}
        rows = [{} for _ in range(N_theta)]
        for entry in term.metadata["terms"]:
            coefficient = _sp.sympify(entry["coefficient"])
            if coefficient.free_symbols:
                raise ValueError(
                    f"BC {term.name!r}: unbound symbols "
                    f"{sorted(map(str, coefficient.free_symbols))} in the coefficient "
                    f"of its {entry['kind']} term; pass their values in params"
                )
            name = entry["unknown"]
            if entry["kind"] == "curl":
                placeholders = {
                    s: _sp.Function(f"{name}_sigma_{label}")(r_sym, t_sym)
                    for s, label in ((-1, "m"), (0, "0"), (+1, "p"))
                }
                for s in (-1, 0, +1):
                    placeholder_to_id[placeholders[s].func] = (name, s)
                formulas = _curl_spin_formulas(cm.m, placeholders)
                scalar_expr += coefficient * sum(
                    unit_coefs[s] * formulas[s] for s in (-1, 0, +1)) / r_sym
                continue
            if entry["kind"] != "cross":
                raise NotImplementedError(
                    f"BC {term.name!r}: unsupported curl-cross term {entry['kind']!r}"
                )
            r_wall = np.full((1, N_theta), grid.r[i_r])
            b_spin = evaluate_vector_background_spin_2d(
                entry["background"], r_wall, grid.theta.reshape(1, -1))
            bm, b0, bp = (b_spin[s].ravel() for s in (-1, 0, +1))
            couplings = {
                (-1, -1): 1j * (-b0), (-1, 0): 1j * bm,
                (0, -1): 1j * bp, (0, +1): 1j * (-bm),
                (+1, 0): 1j * (-bp), (+1, +1): 1j * b0,
            }
            sign = complex(coefficient) * (1 if entry["side"] == "left" else -1)
            offset = cm.layout.field_slice(name).start
            for (s_out, s_in), values in couplings.items():
                u = unit_coefs[s_out]
                if u == 0:
                    continue
                for j in range(N_theta):
                    val = sign * u * values[j]
                    if val == 0:
                        continue
                    col = offset + spin_block[s_in] + j * N_r + i_r
                    rows[j][col] = rows[j].get(col, 0.0) + val
        if scalar_expr != 0:
            curl_rows = self._rows_from_scalar_expr(scalar_expr, placeholder_to_id, i_r)
            for j in range(N_theta):
                for col, val in curl_rows[j].items():
                    rows[j][col] = rows[j].get(col, 0.0) + val
        return rows

    def _combination_rows(self, bct: BCTask, i_r: int) -> List[Dict[int, complex]]:
        """Rows of a BC_LINEAR_COMBINATION: at every wall node the sum
        over its terms of the coefficient there times the term's row."""
        N_theta = self.matrices.grid.N_theta
        rows = [{} for _ in range(N_theta)]
        for coefficient, term in bct.metadata["terms"]:
            c_arr = self._evaluate_at_wall(coefficient, i_r)
            for j, entries in enumerate(self._term_rows(term, i_r)):
                if c_arr[j] == 0:
                    continue
                for col, val in entries.items():
                    rows[j][col] = rows[j].get(col, 0.0) + c_arr[j] * val
        return rows


# Spin coefficients of the unit projections: û·v over the columns
# [σ = −1, 0, +1] of the spin basis v^± = (v_θ ± i v_φ)/√2.
_UNIT_SPIN_COEFFICIENTS = {
    BC_UNIT_R_HAT: {-1: 0.0, 0: 1.0, +1: 0.0},
    BC_UNIT_THETA_HAT: {-1: 1.0 / np.sqrt(2.0), 0: 0.0, +1: 1.0 / np.sqrt(2.0)},
    BC_UNIT_PHI_HAT: {-1: 1j / np.sqrt(2.0), 0: 0.0, +1: -1j / np.sqrt(2.0)},
}

# Spin block hosting the projection on a unit vector.
_UNIT_SIGMA = {BC_UNIT_R_HAT: 0, BC_UNIT_THETA_HAT: +1, BC_UNIT_PHI_HAT: -1}


def _gauge_quadrature_weights(r: np.ndarray, theta: np.ndarray):
    """(w_r, w_theta) for the gauge integral: composite trapezoid on the
    radial nodes and midpoint cells in θ bounded by the inter-node
    midpoints and the poles. Uniform grids use the closed forms so the
    weights match them bit for bit."""
    N_r = len(r)
    if _is_uniform(r):
        w_r = np.full(N_r, float(r[1] - r[0]))
        w_r[0] *= 0.5
        w_r[-1] *= 0.5
    else:
        w_r = np.empty(N_r)
        w_r[0] = 0.5 * (r[1] - r[0])
        w_r[-1] = 0.5 * (r[-1] - r[-2])
        w_r[1:-1] = 0.5 * (r[2:] - r[:-2])
    edges = np.concatenate(([0.0], 0.5 * (theta[1:] + theta[:-1]), [np.pi]))
    w_theta = np.diff(edges)
    if np.allclose(w_theta, theta[1] - theta[0], rtol=1e-12):
        w_theta = np.full(len(theta), float(theta[1] - theta[0]))
    return w_r, w_theta


def _decompose_bc_scalar_expr(scalar_expr, placeholder_to_id: Dict):
    """Split a scalar BC SymPy expression into additive terms of the form
    ``coefficient(r, θ) · action(placeholder)`` with action "pointwise",
    "d_r" (one ∂_r) or "d_theta" (one ∂_θ).

    Returns a list of dicts with keys "angular_coef" (sp.Expr),
    "unknown_name", "sigma" and "action". Higher or mixed derivatives
    raise NotImplementedError; nonlinear terms and terms without a
    placeholder raise ValueError.
    """
    import sympy as _sp
    placeholder_funcs = set(placeholder_to_id.keys())
    if not placeholder_funcs:
        raise ValueError("No placeholders provided")
    expanded = _sp.expand(scalar_expr)
    terms_raw = (
        expanded.args if isinstance(expanded, _sp.Add) else (expanded,)
    )
    out = []
    for term in terms_raw:
        if term == 0:
            continue
        info = _classify_bc_term(term, placeholder_to_id)
        if info is None:
            raise ValueError(
                f"BC term {term} contains no unknown placeholder; the BC "
                f"should have had this absorbed into the RHS value."
            )
        out.append(info)
    return out


def _classify_bc_term(term, placeholder_to_id: Dict):
    """Classify one term as ``coefficient · action(placeholder)``.

    Returns the term dict of `_decompose_bc_scalar_expr`, or None when
    the term contains no placeholder. A term may hold one placeholder
    (linearity) with at most one first derivative.
    """
    import sympy as _sp
    from spdee.ir.fields import r as r_sym, theta as t_sym

    def find_outermost(expr):
        # Outermost placeholder applications: a bare placeholder, or a
        # Derivative whose operand is one. Children of a counted node
        # are not visited, so a placeholder inside Derivative(...) is
        # counted once.
        results = []
        def visit(node):
            if (
                hasattr(node, "func")
                and node.func in placeholder_to_id
                and not isinstance(node, _sp.Derivative)
            ):
                results.append((node, "pointwise", node))
                return
            if isinstance(node, _sp.Derivative):
                inner = node.expr
                if (
                    hasattr(inner, "func")
                    and inner.func in placeholder_to_id
                ):
                    vars_ = node.variables
                    if len(vars_) != 1:
                        raise NotImplementedError(
                            f"BC term {node} has mixed/higher derivative "
                            f"order; only single first-order derivatives "
                            f"of placeholders are supported."
                        )
                    var = vars_[0]
                    if var == r_sym:
                        action = "d_r"
                    elif var == t_sym:
                        action = "d_theta"
                    else:
                        raise NotImplementedError(
                            f"BC term derivative variable {var!r} not r or theta"
                        )
                    results.append((node, action, inner))
                    return
            for arg in getattr(node, "args", ()):
                visit(arg)
        visit(expr)
        return results

    placeholder_appearances = find_outermost(term)
    if not placeholder_appearances:
        return None
    if len(placeholder_appearances) > 1:
        raise ValueError(
            f"BC term {term} contains multiple placeholder appearances: "
            f"{placeholder_appearances}. The BC must be linear in the "
            f"unknown vector spin components."
        )
    sub_expr, action, applied = placeholder_appearances[0]
    unknown_name, sigma = placeholder_to_id[applied.func]
    coef = _sp.simplify(term / sub_expr)
    for fn in coef.atoms(_sp.Function):
        if hasattr(fn, "func") and fn.func in placeholder_to_id:
            raise ValueError(
                f"BC term coefficient {coef} contains a placeholder; "
                f"this term is nonlinear or wasn't fully decomposed."
            )
    return {
        "angular_coef": coef,
        "unknown_name": unknown_name,
        "sigma": sigma,
        "action": action,
    }


def compile_problem(
    system, grid, m: int, order_r: int = 4, order_theta: int = 4,
    prefer_aux_free: bool = True,
    discretization=None,
    params: dict = None,
    coriolis_pairing: str = "center",
    impenetrability_host: str = "auto",
    pole_closure: bool = False,
) -> CompiledProblem:
    """Walk the system, compile matrices, gather BC tasks and return a
    CompiledProblem.

    Parameters
    ----------
    system : EquationSystem
    grid : Grid
    m : int
        Azimuthal mode.
    order_r : int, default 4
        Even formal order ≥ 2 of the radial FD stencils.
    order_theta : int, default 4
        Even formal order ≥ 2 of the angular FD stencils, independent
        of `order_r`.
    prefer_aux_free : bool, default True
        How the composed MHD patterns ``Cross(Curl(unknown), bg)`` and
        ``Curl(Cross(unknown, bg))`` are dispatched. True: the walker
        emits dedicated task kinds built as products of the elementary
        Curl and Cross blocks, with no auxiliary unknowns. False: the
        augmentation pass gives each composed site an auxiliary vector
        unknown and a defining equation.
    pole_closure : bool, default False
        Staggered: add the explicit pole share of the strain's theta-phi
        component, which keeps the pole-adjacent azimuthal velocity ring
        second order at |m| >= 2 but departs from the reference implementation's pole
        treatment (docs/staggered_general_design.md section 2b).
    coriolis_pairing : str, default "center"
        Staggered theta-phi Coriolis pair: "center" (cos theta on the
        centers in both blocks, exactly W-skew), "solar_eigenproblem" (the reference implementation's recipe,
        skew to O(dtheta)) or "adjoint" (edge-cos block with its exact
        adjoint).
    discretization : Discretization or str, optional
        Defaults to COLLOCATED. ``STAGGERED`` returns the staggered
        matrices directly (that path applies its impenetrability BC
        during assembly and uses its own layout, so it bypasses
        CompiledProblem). ``SH_FD``/``DEDALUS`` raise.
    params : dict, optional
        Values bound into symbolic parameters.
    impenetrability_host : str, default "auto"
        Wall row hosting Dot(r̂, v) = 0 on a vector with a divergence
        constraint: 'constraint', 'momentum' or 'auto' (see
        `CompiledProblem`).
    """
    from spdee.backends.fd2.compiler import compile_fd2
    from spdee.backends.fd2.discretization import (
        Discretization, DEFAULT_DISCRETIZATION, require_implemented,
    )
    from spdee.ir.walker import walk_boundary_conditions
    from spdee.ir.augment import augment_equation_system
    method = Discretization.coerce(
        DEFAULT_DISCRETIZATION if discretization is None else discretization
    )
    require_implemented(method)
    if method is Discretization.STAGGERED:
        return compile_fd2(
            system, grid, m, order_r=order_r, order_theta=order_theta,
            discretization=method, params=params, coriolis_pairing=coriolis_pairing,
            pole_closure=pole_closure,
        )
    system = augment_equation_system(system, prefer_aux_free=prefer_aux_free)
    matrices = compile_fd2(
        system, grid, m, order_r=order_r, order_theta=order_theta,
        params=params,
    )
    bc_tasks = walk_boundary_conditions(system, m, composite=True)
    if params:
        from spdee.ir.tasks import bind_bc_parameters
        bc_tasks = bind_bc_parameters(bc_tasks, params)
    bc_tasks = bc_tasks + list(matrices.implicit_bc_tasks)
    return CompiledProblem(
        matrices=matrices, bc_tasks=bc_tasks,
        impenetrability_host=impenetrability_host,
    )
