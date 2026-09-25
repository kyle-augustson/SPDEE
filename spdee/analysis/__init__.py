"""Tools for judging and improving the eigenpairs SPDEE produces.

    conditioning    left eigenvectors, kappa(lambda), first-order sensitivities,
                    eigenvector angles and a ranking of the ill-conditioned modes
    pseudospectra   the epsilon-pseudospectrum on a grid, the resolvent norm
                    along a line, the Kreiss constant
    nonnormality    Henrici's departure from normality, numerical against
                    spectral abscissa, the numerical range, per-mode
                    non-normality
    modes           energy fractions, toroidal/poloidal split and l-content,
                    parity, localisation, residuals; ``identify`` picks an
                    eigenpair by structure
    precondition    equilibration, deflation, a coarse-grid preconditioner for
                    bbjd, parameter continuation and a stall diagnosis
    weights         the weighted inner products everything is measured in

Every module docstring states the mathematics of its quantities and the norm
they are taken in.
"""

from spdee.analysis.weights import Weight, default_weight, natural_weight, mass_support
from spdee.analysis.conditioning import (
    left_eigenvectors, two_sided_dense, condition_numbers, sensitivity, eigenvector_angles,
    condition_report, ConditioningReport,
)
from spdee.analysis.pseudospectra import (
    resolvent_norm, pseudospectrum, resolvent_along_line, kreiss_constant, Pseudospectrum,
)
from spdee.analysis.nonnormality import (
    reduced_operator, ReducedOperator, henrici, numerical_abscissa, numerical_range,
    mode_nonnormality,
)
from spdee.analysis.modes import (
    to_grid_fields, GridFields, features, ModeFeatures, energy_fractions, l_content,
    toroidal_poloidal, parity, localisation, pencil_residual, constraint_residual, ModeTarget,
    Identification, identify,
)
from spdee.analysis.precondition import (
    equilibrate, Equilibration, condition_estimate, Deflation, pressure_null_modes,
    CoarseGridPreconditioner, SchwarzPreconditioner, refine, Refinement, continuation,
    ContinuationPath, diagnose, Diagnosis,
)

__all__ = [
    "Weight", "default_weight", "natural_weight", "mass_support",
    "left_eigenvectors", "two_sided_dense", "condition_numbers", "sensitivity",
    "eigenvector_angles", "condition_report", "ConditioningReport",
    "resolvent_norm", "pseudospectrum", "resolvent_along_line", "kreiss_constant",
    "Pseudospectrum",
    "reduced_operator", "ReducedOperator", "henrici", "numerical_abscissa", "numerical_range",
    "mode_nonnormality",
    "to_grid_fields", "GridFields", "features", "ModeFeatures", "energy_fractions", "l_content",
    "toroidal_poloidal", "parity", "localisation", "pencil_residual", "constraint_residual",
    "ModeTarget", "Identification", "identify",
    "equilibrate", "Equilibration", "condition_estimate", "Deflation", "pressure_null_modes",
    "CoarseGridPreconditioner", "SchwarzPreconditioner", "refine", "Refinement", "continuation",
    "ContinuationPath", "diagnose", "Diagnosis",
]
