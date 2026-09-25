"""The spectrum file contract (spdee.io) and the writer script."""

import os
import subprocess
import sys

import numpy as np

from spdee.io import load_spectrum, spectrum_filename


def test_writer_script_contract(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(root, "scripts", "spdee_spectrum.py"),
           "--core", "sh_fd", "--m", "8", "--nr", "12", "--lmax", "14", "--nu", "2e12",
           "--k", "4", "--fd-order", "2", "--out", str(tmp_path)]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": root, "OMP_NUM_THREADS": "1"},
                   capture_output=True)
    f = tmp_path / spectrum_filename("sh_fd", "hydro", 8, 12, 14, 2e12)
    assert f.exists()
    d = load_spectrum(str(f))
    n = len(d["omega"])
    assert d["omega"].dtype == np.complex128 and d["lam"].shape == (n,)
    assert d["V"].shape[1] == n and d["resid"].shape == (n,)
    assert np.allclose(d["omega"], -1j * d["lam"])
    assert d["resid"].max() < 1e-8
    m = d["meta"]
    for key in ("core", "m", "N_r", "ell_max", "nu_cgs", "delta", "Ek", "Omega0_nHz",
                "omega_convention", "layout", "spdee_commit", "spdee_version", "sigma"):
        assert key in m, key
    assert m["core"] == "sh_fd" and m["m"] == 8 and m["Omega0_nHz"] == 456.0
    # the sectoral Rossby mode is among the k returned, retrograde and damped
    j = int(np.argmin(np.abs(d["omega"] + 2 / 9)))
    assert d["omega"][j].real < 0 and d["omega"][j].imag < 0
