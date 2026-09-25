"""Packaging / installability smoke checks.

The package is pip-installable (metadata present) and a bare ``pytest``
from the repo root collects the curated suite only.  Fast, import-only
checks, no solves.
"""
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_exists_and_declares_core_metadata():
    """A pyproject.toml with name, version, and the core scientific deps must
    exist so ``pip install .`` / editable installs work in CI."""
    pp = REPO_ROOT / "pyproject.toml"
    assert pp.exists(), "pyproject.toml missing — package is not installable"
    meta = tomllib.loads(pp.read_text())
    proj = meta["project"]
    assert proj["name"] == "spdee"
    assert "version" in proj
    deps = " ".join(proj.get("dependencies", []))
    for core in ("numpy", "scipy", "sympy"):
        assert core in deps, f"core dependency {core!r} not declared"


def test_pytest_discovery_is_scoped_to_tests():
    """pytest config must scope discovery to ``tests`` so a bare ``pytest``
    from the root does not walk the solver libraries' own suites or docs."""
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    ini = meta.get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert ini.get("testpaths") == ["tests"], (
        "pyproject [tool.pytest.ini_options] must set testpaths = ['tests']")
    assert "docs" in ini.get("norecursedirs", [])


def test_package_version_matches_pyproject():
    """The importable package version and the packaging metadata agree."""
    import spdee
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert spdee.__version__ == meta["project"]["version"]


def test_core_public_api_importable():
    """The advertised top-level symbolic API imports cleanly (the symbolic
    machinery the reviewer confirmed works in at least some paths)."""
    import spdee
    for name in ("Grad", "Div", "Curl", "Cross", "Dot", "Lap", "Dt",
                 "Equation", "EquationSystem", "Unknown", "Background"):
        assert hasattr(spdee, name), f"public API symbol {name!r} missing"


def test_solar_data_packaged_and_reachable():
    """The tabulated Model S data shipped via package-data is reachable
    relative to the installed package (used by SolarBackgrounds)."""
    import spdee
    data = Path(spdee.__file__).resolve().parent.parent / "data" / "solar"
    assert (data / "model_s_corrected.txt").exists()
