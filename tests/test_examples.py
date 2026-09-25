"""Reduced-resolution smoke of the user-facing examples in ``examples/``.

Each example is imported as a module (their names start with a digit) and its
functions are run at a fraction of the default resolution, asserting the
physics they print at a tolerance loose enough for the coarse grids.  The
figures are not produced here; ``main()`` of each example is what a user runs.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

from spdee.solvers import available

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def load(name):
    """Import ``examples/<name>.py`` as a module."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXAMPLES, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def has_dedalus():
    return importlib.util.find_spec("dedalus") is not None


def test_helmholtz_robin_shell_on_every_core():
    ex = load("01_helmholtz")
    cores = tuple(c for c in ex.CORES if c != "dedalus" or has_dedalus())
    results, pictures = ex.run(walls=("robin",), geometries=("shell",), cores=cores,
                               levels=(0, 1), verbose=False)
    for core in cores:
        targets, values = results[("robin", "shell", core)]
        errs = [abs(v - t) / t for v, t in zip(values[-1], targets)]
        assert max(errs) < 5e-3, (core, errs)
        cp, x, lam = pictures[core]
        th, r, u = ex.eigenfunction(cp, core, x, np.linspace(0.05, np.pi - 0.05, 31))
        assert u.shape == (len(th), len(r))
        assert np.isfinite(u).all()
    assert abs(ex.bessel_rate(1, "dirichlet", "ball") - 4.493409458 ** 2) < 1e-6


def test_inertial_zhang_and_doppler(monkeypatch):
    ex = load("02_inertial_modes")
    monkeypatch.setattr(ex, "N_R", 24)
    monkeypatch.setattr(ex, "ELL_EXTRA", 6)
    _, _, _, rows = ex.zhang_table(1, verbose=False)
    for label, target, nearest_, om_id, fit, _ in rows:
        assert abs(nearest_ - target) < 2e-4, (label, target, nearest_)
        assert abs(om_id - target) < 2e-4, (label, target, om_id)
        assert fit < 0.05, (label, fit)
    for label, omega_i, w1, w2, overlap, _, _ in ex.doppler_check(verbose=False):
        assert abs((w2 - w1) - ex.RATES[1] + ex.RATES[0]) < 1e-9, (label, w1, w2)
        assert overlap > 0.999999, (label, overlap)


def test_convective_onset_staggered():
    ex = load("03_convective_onset")
    results = ex.run(cores=(("staggered", (8, 16)),), verbose=False)
    pencil, scan, Ra_c, lam_c, x = results[("staggered", (8, 16))]
    assert pencil.linearity < 1e-12
    assert abs(Ra_c - 2.2224) < 0.03, Ra_c
    assert abs(lam_c.real) < 1e-6
    assert abs(lam_c.imag - 0.0583) < 3e-3
    th, r, T = ex.temperature_on_grid(pencil, "staggered", x)
    assert T.shape == (len(th), len(r))


@pytest.mark.skipif(not has_dedalus(), reason="Dedalus not installed")
def test_convective_onset_dedalus_agrees():
    ex = load("03_convective_onset")
    results = ex.run(cores=(("dedalus", (8, 16)),), verbose=False)
    _, _, Ra_c, lam_c, _ = results[("dedalus", (8, 16))]
    assert abs(Ra_c - 2.222418) < 1e-2, Ra_c
    assert abs(lam_c.imag - 0.058282) < 1e-3


def test_critical_layer_locate_refine_polish(monkeypatch):
    ex = load("04_critical_layer")
    assert ex.check_rigid_rotation_identity() < 1e-12
    cp_c, K_c, M_c, lam_c, x_c, found = ex.locate(coarse=(10, 48), verbose=False)
    assert abs(-1j * lam_c - (-0.5274 - 0.0192j)) < 5e-3
    assert ex.residual(K_c, M_c, lam_c, x_c) < 1e-10
    cp_f, K_f, M_f, lam_f, x_f, res_f = ex.refine(lam_c, fine=(12, 64), verbose=False)
    assert res_f < 1e-10
    lam_b, x_b, before, after = ex.polish(K_f, M_f, lam_c, ex.interpolate(cp_c, cp_f, x_c),
                                          "interpolated", verbose=False)
    if available()["bbjd"]:
        assert before > 1e-4 and after < 1e-10, (before, after)
        assert abs(lam_b - lam_f) < 1e-8
    else:
        assert after == before, "without the bbjd extension the pair is returned unpolished"
        assert lam_b == lam_c
    om_w, hit = ex.census_window(K_c, M_c, lam_c, found, coarse=(10, 48), verbose=False)
    assert hit.sum() >= 1 and np.abs(om_w - (-1j * lam_c)).min() < 1e-8
    theta_c = ex.critical_colatitude(-1j * lam_b)
    assert abs(np.degrees(theta_c) - 35.7) < 1.0
    assert 0.1 < ex.layer_width(cp_f, x_b, theta_c) < 0.8


def test_critical_layer_convergence(monkeypatch):
    ex = load("04_critical_layer")
    monkeypatch.setattr(ex, "COARSE", (12, 64))
    monkeypatch.setattr(ex, "FINE", (16, 96))
    monkeypatch.setattr(ex, "FINER", (20, 128))
    monkeypatch.setattr(ex, "EK_LADDER", ((1e-4, 16, 96), (3e-5, 16, 144)))
    rows = ex.converge(1j * (-0.52736) + 0.01916, verbose=False)
    om = [row[4] for row in rows]
    assert max(abs(w - om[-1]) for w in om[:3]) < 2e-3
    widths = [row[5] for row in rows if row[0] == "Ek"]
    assert widths[1] < widths[0]


def test_magnetic_decay_spellings(monkeypatch):
    ex = load("05_magnetic_decay")
    monkeypatch.setattr(ex, "RESOLUTION", {"fd2": (12, 18), "staggered": (10, 16),
                                           "sh_fd": (24, 6), "dedalus": (12, 12)})
    cores = ("fd2", "staggered", "sh_fd") + (("dedalus",) if has_dedalus() else ())
    targets, results = ex.run(cores=cores, verbose=False)
    for formulation in ex.FORMULATIONS:
        rec = ex.classification(ex.decay_problem(formulation))
        if formulation == "b":
            assert rec[True].involutions == {"b": "induction"}
            assert rec[False].divergence_cleaning["b"].multiplier == "__div_b"
        elif formulation == "A":
            assert rec[True].multipliers == {"psi": "coulomb"}
        for core in cores:
            got = results[(formulation, core)]
            if formulation == "b_free" and core == "staggered":
                assert got is None
                continue
            _, _, _, rates, divs, _ = got
            errs = [abs(g - t) / t for g, (_, t) in zip(rates, targets)]
            assert max(errs) < 3e-2, (formulation, core, errs)
            if formulation == "b" and core in ("staggered", "dedalus"):
                assert max(divs) < 1e-9, (core, divs)
    assert abs(ex.gradient_rates()[0] - 3.386587) < 1e-5
