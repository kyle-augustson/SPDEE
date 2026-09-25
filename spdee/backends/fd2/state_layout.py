"""spdee.backends.fd2.state_layout — system-wide state vector layout.

An equation system has several unknowns (scalars, vectors, tensors)
that are concatenated into the single state vector `x` of the
generalized eigenvalue problem `A x = lambda B x`. This module fixes
the ordering: given the primary unknowns (declared in the IR) and any
auxiliary intermediates (introduced by the compiler for state
augmentation, e.g. `w = Curl(v)`), it computes the total size, the
slice of every field, and pack / unpack helpers between
`{name: 2D-or-dict-of-2D}` and the flat vector.

Primaries come first, then auxiliaries. Inside each field block the
single-field layout of layout.py applies: one r-major (N_r, N_theta)
block per spin component, 3 for a vector and 9 for a tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from spdee.backends.fd2.layout import (
    flatten_2d, reshape_2d, state_size,
    vector_component_slice, tensor_component_slice,
    split_vector_state, split_tensor_state,
    assemble_vector_state, assemble_tensor_state,
)

ROLE_PRIMARY = "primary"
ROLE_AUXILIARY = "auxiliary"
_VALID_KINDS = ("scalar", "vector", "tensor")


@dataclass(frozen=True)
class FieldEntry:
    """One field in the system state.

    Attributes
    ----------
    name : str
        Lookup name, unique within the StateLayout.
    kind : str
        "scalar" / "vector" / "tensor".
    role : str
        ROLE_PRIMARY or ROLE_AUXILIARY.
    offset : int
        Start of this field's block in the flat state.
    size : int
        Length of the block, COMPONENTS_PER_KIND[kind] * N_r * N_theta.
    """
    name: str
    kind: str
    role: str
    offset: int
    size: int

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.offset + self.size)


class StateLayout:
    """Ordered layout of all fields in a system-wide state vector.

    Parameters
    ----------
    primary : list of (name, kind)
        User-declared unknowns, in their canonical order.
    N_r, N_theta : int
        Grid dimensions; slice sizes depend on them.
    auxiliary : list of (name, kind), optional
        Compiler-introduced intermediates, appended after the primaries.

    Names must be unique across both lists.
    """

    def __init__(
        self,
        primary: List[Tuple[str, str]],
        N_r: int,
        N_theta: int,
        auxiliary: Optional[List[Tuple[str, str]]] = None,
    ):
        if N_r <= 0 or N_theta <= 0:
            raise ValueError(f"Invalid grid: N_r={N_r}, N_theta={N_theta}")
        self._N_r = N_r
        self._N_theta = N_theta
        self._block = N_r * N_theta
        auxiliary = list(auxiliary) if auxiliary else []
        entries: List[FieldEntry] = []
        names_seen: set = set()
        offset = 0
        for role, items in [
            (ROLE_PRIMARY, primary),
            (ROLE_AUXILIARY, auxiliary),
        ]:
            for spec in items:
                if not isinstance(spec, tuple) or len(spec) != 2:
                    raise ValueError(
                        f"Each field spec must be (name, kind); got {spec!r}"
                    )
                name, kind = spec
                if not isinstance(name, str) or not name:
                    raise ValueError(f"Invalid field name: {name!r}")
                if kind not in _VALID_KINDS:
                    raise ValueError(
                        f"Invalid kind {kind!r} for field {name!r}; "
                        f"must be one of {_VALID_KINDS}"
                    )
                if name in names_seen:
                    raise ValueError(
                        f"Duplicate field name {name!r} in StateLayout"
                    )
                names_seen.add(name)
                size = state_size(kind, N_r, N_theta)
                entries.append(FieldEntry(
                    name=name, kind=kind, role=role,
                    offset=offset, size=size,
                ))
                offset += size
        self._entries = entries
        self._by_name: Dict[str, FieldEntry] = {e.name: e for e in entries}
        self._total_size = offset

    @property
    def N_r(self) -> int:
        return self._N_r

    @property
    def N_theta(self) -> int:
        return self._N_theta

    @property
    def size(self) -> int:
        """Total length of the flat state vector."""
        return self._total_size

    @property
    def primary_names(self) -> List[str]:
        """Names of primary unknowns, in canonical order."""
        return [e.name for e in self._entries if e.role == ROLE_PRIMARY]

    @property
    def auxiliary_names(self) -> List[str]:
        """Names of auxiliary intermediates, in canonical order."""
        return [e.name for e in self._entries if e.role == ROLE_AUXILIARY]

    @property
    def all_names(self) -> List[str]:
        """All field names, primary then auxiliary."""
        return [e.name for e in self._entries]

    @property
    def primary_size(self) -> int:
        """Total size of the primary block."""
        return sum(e.size for e in self._entries if e.role == ROLE_PRIMARY)

    def entry(self, name: str) -> FieldEntry:
        if name not in self._by_name:
            raise KeyError(
                f"Field {name!r} not in StateLayout. Known: "
                f"{list(self._by_name.keys())}"
            )
        return self._by_name[name]

    def field_slice(self, name: str) -> slice:
        """Slice of the named field's full block in the flat state."""
        return self.entry(name).slice

    def component_slice(
        self, name: str, component: Union[int, Tuple[int, int]]
    ) -> slice:
        """Slice of one spin component of a vector (int) or tensor
        ((int, int)) field, in system-wide coordinates.
        """
        entry = self.entry(name)
        if entry.kind == "scalar":
            raise ValueError(
                f"Field {name!r} is scalar; component_slice does not apply. "
                f"Use field_slice() instead."
            )
        if entry.kind == "vector":
            inner = vector_component_slice(component, self._N_r, self._N_theta)
        elif entry.kind == "tensor":
            inner = tensor_component_slice(component, self._N_r, self._N_theta)
        else:
            raise AssertionError(f"unhandled kind {entry.kind}")
        return slice(entry.offset + inner.start, entry.offset + inner.stop)

    def pack(
        self,
        values: Dict[str, Union[np.ndarray, Dict]],
    ) -> np.ndarray:
        """Build a flat complex state vector from per-field data.

        `values` maps field names (primary or auxiliary) to a
        (N_r, N_theta) array for scalars, {sigma: array} for vectors,
        or {(s1, s2): array} for tensors. Missing fields are zero.
        """
        out = np.zeros(self._total_size, dtype=complex)
        for entry in self._entries:
            if entry.name not in values:
                continue
            data = values[entry.name]
            if entry.kind == "scalar":
                if not isinstance(data, np.ndarray):
                    raise TypeError(
                        f"Field {entry.name!r}: scalar expected np.ndarray, "
                        f"got {type(data).__name__}"
                    )
                out[entry.slice] = flatten_2d(data)
            elif entry.kind == "vector":
                if not isinstance(data, dict):
                    raise TypeError(
                        f"Field {entry.name!r}: vector expected dict "
                        f"{{sigma: array}}, got {type(data).__name__}"
                    )
                out[entry.slice] = assemble_vector_state(
                    data, self._N_r, self._N_theta
                )
            elif entry.kind == "tensor":
                if not isinstance(data, dict):
                    raise TypeError(
                        f"Field {entry.name!r}: tensor expected dict "
                        f"{{(s1, s2): array}}, got {type(data).__name__}"
                    )
                out[entry.slice] = assemble_tensor_state(
                    data, self._N_r, self._N_theta
                )
            else:
                raise AssertionError(f"unhandled kind {entry.kind}")
        return out

    def unpack(
        self,
        flat: np.ndarray,
        role: Optional[str] = None,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """Split a flat state vector into per-field data.

        With `role` "primary" or "auxiliary" only that group is
        returned; None returns every field. Scalars come back as
        (N_r, N_theta) arrays, vectors and tensors as component dicts.
        """
        if flat.size != self._total_size:
            raise ValueError(
                f"flat state has {flat.size} elements; expected "
                f"{self._total_size}"
            )
        out: Dict[str, Union[np.ndarray, Dict]] = {}
        for entry in self._entries:
            if role is not None and entry.role != role:
                continue
            block = flat[entry.slice]
            if entry.kind == "scalar":
                out[entry.name] = reshape_2d(block, self._N_r, self._N_theta)
            elif entry.kind == "vector":
                out[entry.name] = split_vector_state(
                    block, self._N_r, self._N_theta
                )
            elif entry.kind == "tensor":
                out[entry.name] = split_tensor_state(
                    block, self._N_r, self._N_theta
                )
            else:
                raise AssertionError(f"unhandled kind {entry.kind}")
        return out

    def __repr__(self) -> str:
        parts = []
        for e in self._entries:
            tag = "P" if e.role == ROLE_PRIMARY else "A"
            parts.append(f"{e.name}({e.kind}, {tag}, {e.size})")
        return (
            f"StateLayout(N_r={self._N_r}, N_theta={self._N_theta}, "
            f"size={self._total_size}, [{', '.join(parts)}])"
        )
