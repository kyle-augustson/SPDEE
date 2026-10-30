"""The maintained FPLUS and BBJD libraries through ``spdee.solvers.external``:
selection rules everywhere, and, where SPDEE_FPLUS_PATH / SPDEE_BBJD_PATH
name a build, the same eigenvalues as the reference ARPACK + SuperLU path.
"""

import numpy as np
import pytest

from spdee import solvers
from spdee.solvers import external

from test_solvers import K, M, SIGMA, KK, _check

EXT = external.describe()
HAVE_FPLUS = EXT["fplus"]["ok"]
HAVE_BBJD = EXT["bbjd"]["ok"]


def _no_env(monkeypatch, pkg):
    for var in external._ENV[pkg]:
        monkeypatch.delenv(var, raising=False)


def test_external_needs_a_path(monkeypatch):
    _no_env(monkeypatch, "FPLUS")
    _no_env(monkeypatch, "BBJD")
    with pytest.raises(external.ExternalLibraryError, match="SPDEE_FPLUS_PATH"):
        external.load_fplus("external")
    with pytest.raises(external.ExternalLibraryError, match="SPDEE_BBJD_PATH"):
        external.load_bbjd("external")
    assert external.load_fplus("auto")[1] == "vendored"


def test_path_without_package(monkeypatch, tmp_path):
    monkeypatch.setenv("SPDEE_FPLUS_PATH", str(tmp_path))
    with pytest.raises(external.ExternalLibraryError, match="__init__.py"):
        external.load_fplus("auto")


def test_unknown_library():
    with pytest.raises(ValueError):
        external.load_fplus("nope")


def test_vendored_refuses_library_options():
    if not solvers.available().get("fplus"):
        pytest.skip("FPLUS copy not importable")
    P = solvers._shifted(K, M, SIGMA)
    with pytest.raises(TypeError, match="SPDEE_FPLUS_PATH"):
        solvers.factorize(P, "fplus", library="vendored", relax_rows=32)


@pytest.mark.skipif(not HAVE_FPLUS, reason="SPDEE_FPLUS_PATH does not name a loadable FPLUS build")
@pytest.mark.parametrize("pivot", ["partial", "rook", "full"])
def test_fplus_external(pivot):
    lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="fplus",
                                  factor_options=dict(library="external", pivot_strategy=pivot))
    _check(lam)


@pytest.mark.skipif(not HAVE_FPLUS, reason="SPDEE_FPLUS_PATH does not name a loadable FPLUS build")
@pytest.mark.parametrize("relax,panel", [(-1, -1), (32, 512), (0, 1)])
def test_fplus_external_settings(relax, panel):
    """Automatic, the paper's production setting, and exact supernodes."""
    P = solvers._shifted(K, M, SIGMA)
    fac = solvers.factorize(P, "fplus", library="external", relax_rows=relax, panel=panel)
    assert fac.library == "external"
    b = np.random.default_rng(1).standard_normal(P.shape[0]) + 0j
    Pc = P.tocsr()
    assert np.linalg.norm(Pc @ fac.solve(b) - b) < 1e-10 * np.linalg.norm(b)
    assert np.linalg.norm(Pc.conj().T @ fac.solve_H(b) - b) < 1e-10 * np.linalg.norm(b)


@pytest.mark.skipif(not HAVE_FPLUS, reason="SPDEE_FPLUS_PATH does not name a loadable FPLUS build")
def test_fplus_auto_takes_external():
    P = solvers._shifted(K, M, SIGMA)
    assert solvers.factorize(P, "fplus").library == "external"
    assert solvers.factorize(P, "fplus", library="vendored").library == "vendored"
    assert solvers.available()["fplus_external"] == EXT["fplus"]["path"]


@pytest.mark.skipif(not HAVE_BBJD, reason="SPDEE_BBJD_PATH does not name a loadable BBJD build")
@pytest.mark.parametrize("factor", ["scipy", "fplus"])
def test_bbjd_external(factor):
    info = {}
    lam, V = solvers.shift_invert(K, M, SIGMA, KK, method="bbjd", factor=factor,
                                  return_vectors=True, maxsub=60, maxiter=400,
                                  library="external", info=info)
    assert info["library"] == "external"
    _check(lam, rtol=1e-7)
    for j in range(len(lam)):
        x = V[:, j]
        r = np.linalg.norm(K @ x - lam[j] * (M @ x)) / (np.linalg.norm(K @ x) + 1e-300)
        assert r < 1e-7
