"""spdee.backends.common — utilities shared across backends.

field_eval evaluates background fields on numerical grids.
"""

from spdee.backends.common.field_eval import (
    evaluate_scalar_background_2d,
    evaluate_vector_background_spin_2d,
)

__all__ = [
    "evaluate_scalar_background_2d",
    "evaluate_vector_background_spin_2d",
]
