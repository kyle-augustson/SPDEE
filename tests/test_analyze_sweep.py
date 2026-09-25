"""Tests for `scripts/analyze_sweep.py`.

The analyzer is a CLI tool, so testing focuses on its programmatic
internals (`_detect_sweep_kind`, `_analyze_ball`, `_analyze_le`)
rather than end-to-end CLI behavior. We synthesize tiny mock
record dictionaries with the schema produced by the two sweep
scripts and verify the analyzer reports what we expect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts dir to path so we can import analyze_sweep.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import analyze_sweep as az  # noqa: E402


# ---------------------------------------------------------------------------
# Sweep-kind detection
# ---------------------------------------------------------------------------

def test_detect_le_when_Le_key_present():
    """A record with `Le` → Le sweep."""
    rec = {"Le": 0.05, "m": 3, "omega": np.array([-0.5]), "gamma": np.array([-0.1])}
    assert az._detect_sweep_kind([rec]) == "le"


def test_detect_ball_when_no_Le_key():
    """A record without `Le` → ball sweep."""
    rec = {"m": 3, "N_r": 256, "N_theta": 128,
           "omega": np.array([-0.5]), "gamma": np.array([-0.1])}
    assert az._detect_sweep_kind([rec]) == "ball"


def test_detect_empty():
    assert az._detect_sweep_kind([]) == "empty"


# ---------------------------------------------------------------------------
# Ball-sweep analysis
# ---------------------------------------------------------------------------

def _mock_ball_record(m, N_r, N_theta, omega, gamma,
                     t_lu=10.0, t_arpack=5.0, solve_time=15.0,
                     rss_kb=1024*1024):
    """Build a minimal ball-sweep record dict matching the .npz schema."""
    return {
        "m": m, "N_r": N_r, "N_theta": N_theta,
        "omega": np.array([omega]), "gamma": np.array([gamma]),
        "eigenvalues": np.array([complex(-gamma, -omega)]),
        "target_omega": -2.0 / (m + 1),
        "t_lu_decompose": t_lu,
        "t_arpack": t_arpack,
        "solve_time": solve_time,
        "rss_peak_kb": rss_kb,
    }


def test_ball_analysis_basic():
    """Three cells × three m's → expected structure."""
    records = []
    for cell in [(48, 24), (48, 48), (96, 24)]:
        for m in [1, 2, 3]:
            # Make ω track -2/(m+1) approximately.
            records.append(_mock_ball_record(
                m, cell[0], cell[1],
                omega=-2.0/(m+1) + 0.01, gamma=-0.05,
            ))
    s = az._analyze_ball(records)
    assert s["kind"] == "ball"
    assert s["cells"] == [(48, 24), (48, 48), (96, 24)]
    assert s["m_values"] == [1, 2, 3]
    assert len(s["rows"]) == 3
    # Each row should have ω and γ for each cell.
    for row in s["rows"]:
        for cell in s["cells"]:
            assert f"ω_{cell}" in row
            assert f"γ_{cell}" in row


def test_ball_mode_jump_detection():
    """A 0.05 jump in ω between two cells should be flagged."""
    records = [
        _mock_ball_record(3, 48, 48, omega=-0.5, gamma=-0.1),
        _mock_ball_record(3, 96, 48, omega=-0.45, gamma=-0.1),
    ]
    s = az._analyze_ball(records)
    # Single cell pair (48,48) → (96,48), N_r axis.
    assert len(s["convergence"]) == 1
    cmp = s["convergence"][0]
    assert cmp["axis"] == "N_r"
    assert cmp["per_m"][0]["mode_jump"] is True
    assert cmp["per_m"][0]["d_omega"] == pytest.approx(0.05, abs=1e-10)


def test_ball_no_mode_jump_for_small_delta():
    """A 1e-5 difference should not flag."""
    records = [
        _mock_ball_record(3, 48, 48, omega=-0.5, gamma=-0.1),
        _mock_ball_record(3, 96, 48, omega=-0.50001, gamma=-0.1),
    ]
    s = az._analyze_ball(records)
    assert s["convergence"][0]["per_m"][0]["mode_jump"] is False


def test_ball_three_pair_comparison():
    """3 cells → 3 pairs (C(3,2))."""
    records = []
    for cell in [(48, 24), (48, 48), (96, 24)]:
        records.append(_mock_ball_record(1, cell[0], cell[1],
                                         omega=-1.0, gamma=-0.01))
    s = az._analyze_ball(records)
    assert len(s["convergence"]) == 3
    axes = {c["axis"] for c in s["convergence"]}
    # 3 cells: (48,24) → (48,48) N_θ; (48,24) → (96,24) N_r; (48,48) → (96,24) both.
    assert axes == {"N_r", "N_θ", "both"}


# ---------------------------------------------------------------------------
# Le-sweep analysis
# ---------------------------------------------------------------------------

def _mock_le_record(m, Le, omega, gamma, *,
                   N_r=48, N_theta=48, Ek=3e-3, Pm=1.0,
                   amplitude_ratio=30000.0,
                   r_inner=0.71, r_outer=0.985,
                   t_lu=10.0, t_arpack=5.0, solve_time=15.0,
                   rss_kb=1024*1024):
    """Build a minimal Le-sweep record dict matching the .npz schema."""
    return {
        "Le": Le, "m": m, "Ek": Ek, "Pm": Pm,
        "amplitude_ratio": amplitude_ratio,
        "N_r": N_r, "N_theta": N_theta,
        "r_inner": r_inner, "r_outer": r_outer,
        "omega": np.array([omega]), "gamma": np.array([gamma]),
        "eigenvalues": np.array([complex(-gamma, -omega)]),
        "rossby_omega": omega, "rossby_gamma": gamma,
        "target_omega": -2.0 / (m + 1),
        "t_lu_decompose": t_lu,
        "t_arpack": t_arpack,
        "solve_time": solve_time,
        "rss_peak_kb": rss_kb,
    }


def test_le_analysis_basic():
    """Verify the Le-sweep analyzer builds the expected structure."""
    # Manufactured Le² scaling: Δω = 0.2 · Le².
    records = []
    for Le in [0.0, 0.01, 0.03, 0.1]:
        omega = -0.5 + 0.2 * Le**2
        records.append(_mock_le_record(3, Le, omega, -0.1))
    s = az._analyze_le(records)
    assert s["kind"] == "le"
    assert s["m_values"] == [3]
    assert s["Le_values"] == [0.0, 0.01, 0.03, 0.1]
    assert len(s["per_m"]) == 1
    entry = s["per_m"][0]
    assert entry["m"] == 3
    assert entry["anchor"] == 0.0


def test_le_scaling_law_extraction():
    """|Δω|/Le² should equal 0.2 to high precision for the manufactured data."""
    records = []
    for Le in [0.0, 0.01, 0.03, 0.1]:
        omega = -0.5 + 0.2 * Le**2
        records.append(_mock_le_record(3, Le, omega, -0.1))
    s = az._analyze_le(records)
    for row in s["per_m"][0]["scaling"]:
        if row["Le"] > 0:
            assert row["dom_over_le2"] == pytest.approx(0.2, abs=1e-12)


def test_le_anchor_fallback_to_smallest():
    """If Le=0 isn't present, anchor at the smallest Le."""
    records = []
    for Le in [0.01, 0.05]:
        records.append(_mock_le_record(3, Le, -0.5, -0.1))
    s = az._analyze_le(records)
    assert s["per_m"][0]["anchor"] == 0.01


def test_le_multi_m():
    """Multiple m values should be handled independently."""
    records = []
    for m in [1, 2, 3]:
        for Le in [0.0, 0.05]:
            records.append(_mock_le_record(m, Le, -1.0/(m+1), -0.1))
    s = az._analyze_le(records)
    assert s["m_values"] == [1, 2, 3]
    assert len(s["per_m"]) == 3


# ---------------------------------------------------------------------------
# Text and markdown formatting smoke tests
# ---------------------------------------------------------------------------

def test_format_le_text_no_crash():
    """Formatter should produce text for an Le-sweep summary."""
    records = [_mock_le_record(3, Le, -0.5 + 0.2*Le**2, -0.1)
               for Le in [0.0, 0.01, 0.1]]
    s = az._analyze_le(records)
    text = az._format_le_text(s)
    assert "Le sweep" in text
    assert "m = 3" in text


def test_format_ball_text_no_crash():
    records = []
    for cell in [(48, 24), (48, 48)]:
        for m in [1, 2]:
            records.append(_mock_ball_record(m, cell[0], cell[1], -0.5, -0.1))
    s = az._analyze_ball(records)
    text = az._format_ball_text(s)
    assert "Ball sweep" in text
    assert "ω" in text


def test_format_le_markdown_no_crash():
    records = [_mock_le_record(3, Le, -0.5 + 0.2*Le**2, -0.1)
               for Le in [0.0, 0.01, 0.1]]
    s = az._analyze_le(records)
    md = az._format_le_markdown(s, Path("/tmp/test"))
    assert "# Le-sweep" in md


def test_format_ball_markdown_no_crash():
    records = []
    for cell in [(48, 24), (48, 48)]:
        for m in [1, 2]:
            records.append(_mock_ball_record(m, cell[0], cell[1], -0.5, -0.1))
    s = az._analyze_ball(records)
    md = az._format_ball_markdown(s, Path("/tmp/test"))
    assert "# Ball sweep" in md
