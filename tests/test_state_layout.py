"""Tests for spdee.backends.fd2.state_layout.StateLayout."""

import numpy as np
import pytest

from spdee.backends.fd2.layout import (
    reshape_2d,
    SPIN_COMPONENTS_VECTOR, SPIN_COMPONENTS_TENSOR,
)
from spdee.backends.fd2.state_layout import (
    StateLayout, ROLE_PRIMARY, ROLE_AUXILIARY,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_simple_layout_scalar_only():
    """A single-scalar layout."""
    layout = StateLayout(
        primary=[("p", "scalar")],
        N_r=10, N_theta=8,
    )
    assert layout.size == 80  # 1 * 10 * 8
    assert layout.primary_names == ["p"]
    assert layout.auxiliary_names == []
    e = layout.entry("p")
    assert e.offset == 0
    assert e.size == 80
    assert e.kind == "scalar"
    assert e.role == ROLE_PRIMARY


def test_layout_vector_only():
    """A single-vector layout has 3 spin blocks of N_r * N_theta each."""
    layout = StateLayout(
        primary=[("v", "vector")],
        N_r=10, N_theta=8,
    )
    assert layout.size == 240  # 3 * 80
    e = layout.entry("v")
    assert e.kind == "vector"
    assert e.size == 240


def test_layout_tensor_only():
    """A single-tensor layout has 9 spin blocks."""
    layout = StateLayout(
        primary=[("T", "tensor")],
        N_r=10, N_theta=8,
    )
    assert layout.size == 720  # 9 * 80
    e = layout.entry("T")
    assert e.kind == "tensor"
    assert e.size == 720


def test_layout_multiple_fields_ordering():
    """Fields are stored in the given order; offsets accumulate."""
    layout = StateLayout(
        primary=[("v", "vector"), ("p", "scalar")],
        N_r=4, N_theta=3,
    )
    block = 12  # N_r * N_theta
    assert layout.entry("v").offset == 0
    assert layout.entry("v").size == 3 * block
    assert layout.entry("p").offset == 3 * block
    assert layout.entry("p").size == block
    assert layout.size == 4 * block


def test_layout_with_auxiliaries():
    """Auxiliaries are appended after primaries."""
    layout = StateLayout(
        primary=[("v", "vector"), ("p", "scalar")],
        auxiliary=[("w", "vector")],
        N_r=4, N_theta=3,
    )
    block = 12
    assert layout.primary_names == ["v", "p"]
    assert layout.auxiliary_names == ["w"]
    # w comes after v (3 blocks) + p (1 block) = 4 blocks
    assert layout.entry("w").offset == 4 * block
    assert layout.entry("w").role == ROLE_AUXILIARY
    # primary_size excludes auxiliaries
    assert layout.primary_size == 4 * block
    assert layout.size == 7 * block  # plus w's 3 blocks


def test_layout_duplicate_name_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        StateLayout(primary=[("p", "scalar"), ("p", "vector")], N_r=4, N_theta=3)


def test_layout_duplicate_across_primary_and_auxiliary():
    """A name can't appear in both primary and auxiliary."""
    with pytest.raises(ValueError, match="Duplicate"):
        StateLayout(
            primary=[("v", "vector")],
            auxiliary=[("v", "scalar")],
            N_r=4, N_theta=3,
        )


def test_layout_invalid_kind():
    with pytest.raises(ValueError, match="Invalid kind"):
        StateLayout(primary=[("p", "bogus")], N_r=4, N_theta=3)


def test_layout_bad_grid():
    with pytest.raises(ValueError):
        StateLayout(primary=[("p", "scalar")], N_r=0, N_theta=3)


# ---------------------------------------------------------------------------
# Field and component slicing
# ---------------------------------------------------------------------------

def test_field_slice_basic():
    layout = StateLayout(
        primary=[("v", "vector"), ("p", "scalar")],
        N_r=4, N_theta=3,
    )
    block = 12
    assert layout.field_slice("v") == slice(0, 3 * block)
    assert layout.field_slice("p") == slice(3 * block, 4 * block)


def test_component_slice_vector():
    layout = StateLayout(
        primary=[("p", "scalar"), ("v", "vector")],
        N_r=4, N_theta=3,
    )
    block = 12
    # v starts at offset block (after p)
    # v's sigma=-1 block is the first of v's 3 sub-blocks
    sl = layout.component_slice("v", -1)
    assert sl == slice(block + 0, block + block)
    sl = layout.component_slice("v", 0)
    assert sl == slice(block + block, block + 2 * block)
    sl = layout.component_slice("v", +1)
    assert sl == slice(block + 2 * block, block + 3 * block)


def test_component_slice_tensor():
    layout = StateLayout(primary=[("T", "tensor")], N_r=2, N_theta=2)
    block = 4
    # T^{0,0} is index 4 in lex order ((-, -), (-, 0), (-, +), (0, -), (0, 0), ...)
    expected_idx = SPIN_COMPONENTS_TENSOR.index((0, 0))
    sl = layout.component_slice("T", (0, 0))
    assert sl == slice(expected_idx * block, (expected_idx + 1) * block)


def test_component_slice_rejects_scalar():
    layout = StateLayout(primary=[("p", "scalar")], N_r=2, N_theta=2)
    with pytest.raises(ValueError, match="scalar"):
        layout.component_slice("p", 0)


def test_field_slice_unknown_name():
    layout = StateLayout(primary=[("p", "scalar")], N_r=2, N_theta=2)
    with pytest.raises(KeyError):
        layout.field_slice("nope")


# ---------------------------------------------------------------------------
# Pack and unpack
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip_scalar():
    layout = StateLayout(primary=[("p", "scalar")], N_r=4, N_theta=3)
    p_arr = (np.arange(12) + 1j * np.arange(12)).reshape(4, 3)
    flat = layout.pack({"p": p_arr})
    out = layout.unpack(flat)
    assert np.allclose(out["p"], p_arr)


def test_pack_unpack_roundtrip_vector():
    layout = StateLayout(primary=[("v", "vector")], N_r=4, N_theta=3)
    components = {
        sigma: (sigma + 2) * np.ones((4, 3), dtype=complex)
        for sigma in SPIN_COMPONENTS_VECTOR
    }
    flat = layout.pack({"v": components})
    out = layout.unpack(flat)
    for sigma in SPIN_COMPONENTS_VECTOR:
        assert np.allclose(out["v"][sigma], components[sigma])


def test_pack_unpack_roundtrip_tensor():
    layout = StateLayout(primary=[("T", "tensor")], N_r=4, N_theta=3)
    components = {
        key: (i + 1.0) * np.ones((4, 3), dtype=complex)
        for i, key in enumerate(SPIN_COMPONENTS_TENSOR)
    }
    flat = layout.pack({"T": components})
    out = layout.unpack(flat)
    for key in SPIN_COMPONENTS_TENSOR:
        assert np.allclose(out["T"][key], components[key])


def test_pack_unpack_multifield():
    """Pack and unpack a system with multiple unknowns."""
    layout = StateLayout(
        primary=[("v", "vector"), ("p", "scalar")],
        N_r=4, N_theta=3,
    )
    v_components = {sigma: float(sigma + 2) * np.ones((4, 3), dtype=complex)
                    for sigma in SPIN_COMPONENTS_VECTOR}
    p_arr = np.arange(12, dtype=complex).reshape(4, 3)
    flat = layout.pack({"v": v_components, "p": p_arr})
    out = layout.unpack(flat)
    for sigma in SPIN_COMPONENTS_VECTOR:
        assert np.allclose(out["v"][sigma], v_components[sigma])
    assert np.allclose(out["p"], p_arr)


def test_pack_missing_field_defaults_to_zero():
    """Unspecified fields become zeros."""
    layout = StateLayout(
        primary=[("v", "vector"), ("p", "scalar")],
        N_r=4, N_theta=3,
    )
    # Only specify p, leave v unset
    p_arr = np.ones((4, 3), dtype=complex)
    flat = layout.pack({"p": p_arr})
    out = layout.unpack(flat)
    for sigma in SPIN_COMPONENTS_VECTOR:
        assert np.allclose(out["v"][sigma], 0)
    assert np.allclose(out["p"], 1)


def test_unpack_by_role_primary():
    """Unpack only primaries."""
    layout = StateLayout(
        primary=[("v", "vector")],
        auxiliary=[("w", "vector")],
        N_r=4, N_theta=3,
    )
    flat = np.zeros(layout.size, dtype=complex)
    out = layout.unpack(flat, role=ROLE_PRIMARY)
    assert "v" in out
    assert "w" not in out


def test_unpack_by_role_auxiliary():
    layout = StateLayout(
        primary=[("v", "vector")],
        auxiliary=[("w", "vector")],
        N_r=4, N_theta=3,
    )
    flat = np.zeros(layout.size, dtype=complex)
    out = layout.unpack(flat, role=ROLE_AUXILIARY)
    assert "w" in out
    assert "v" not in out


def test_pack_wrong_type_raises():
    layout = StateLayout(primary=[("v", "vector")], N_r=4, N_theta=3)
    # vector expects a dict, not an array
    with pytest.raises(TypeError, match="dict"):
        layout.pack({"v": np.zeros((4, 3))})


def test_pack_scalar_wrong_type_raises():
    layout = StateLayout(primary=[("p", "scalar")], N_r=4, N_theta=3)
    with pytest.raises(TypeError, match="np.ndarray"):
        layout.pack({"p": {-1: 1, 0: 2, 1: 3}})


def test_unpack_wrong_size_raises():
    layout = StateLayout(primary=[("p", "scalar")], N_r=4, N_theta=3)
    with pytest.raises(ValueError, match="elements"):
        layout.unpack(np.zeros(5))


# ---------------------------------------------------------------------------
# Component slicing values
# ---------------------------------------------------------------------------

def test_component_slice_extracts_correct_data():
    """Use component_slice on a packed state and verify we get back the
    same data we put in."""
    layout = StateLayout(
        primary=[("p", "scalar"), ("v", "vector")],
        N_r=4, N_theta=3,
    )
    v_components = {
        -1: 1.5 * np.ones((4, 3), dtype=complex),
         0: 2.5 * np.ones((4, 3), dtype=complex),
        +1: 3.5 * np.ones((4, 3), dtype=complex),
    }
    p_arr = np.ones((4, 3), dtype=complex) * 7.0
    flat = layout.pack({"p": p_arr, "v": v_components})

    # Extract v's sigma=0 component
    sl = layout.component_slice("v", 0)
    v0_flat = flat[sl]
    v0_arr = reshape_2d(v0_flat, 4, 3)
    assert np.allclose(v0_arr, 2.5)

    # Extract v's sigma=-1 component
    sl = layout.component_slice("v", -1)
    vm_arr = reshape_2d(flat[sl], 4, 3)
    assert np.allclose(vm_arr, 1.5)


# ---------------------------------------------------------------------------
# repr is informative
# ---------------------------------------------------------------------------

def test_repr_is_informative():
    layout = StateLayout(
        primary=[("v", "vector")],
        auxiliary=[("w", "vector")],
        N_r=4, N_theta=3,
    )
    s = repr(layout)
    assert "v" in s and "w" in s
    assert "vector" in s
    assert "N_r=4" in s
