"""Pseudospectra and non-normality (spdee.analysis.pseudospectra,
spdee.analysis.nonnormality) against known answers: the resolvent norm of a
Hermitian pencil is 1/dist(z, spectrum) so its pseudospectra are discs
(all three methods, the projected one a lower bound); a 2 x 2 near-Jordan
block has the closed-form resolvent norm, Henrici departure and numerical
range; the per-mode non-normality is 1 for a normal operator and kappa for
the Jordan block; the Kreiss constant of a damped normal operator is the
known supremum along the growth half-plane.
"""

import numpy as np
import scipy.sparse as sp

from spdee import analysis as an


def _hermitian(n=40, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return sp.csr_matrix(A + A.conj().T)


def _grid(lam0):
    return np.linspace(lam0 - 1, lam0 + 1, 7) + 0.0137, np.linspace(-1, 1, 7) + 0.0171


def test_hermitian_pseudospectra_are_discs():
    K = _hermitian()
    lam, X = np.linalg.eigh(K.toarray())
    re, im = _grid(lam[0])
    Z = re[None, :] + 1j * im[:, None]
    dist = np.min(np.abs(Z[..., None] - lam[None, None, :]), axis=-1)
    ps_lu = an.pseudospectrum(K, None, re, im, method="lu")
    ps_dense = an.pseudospectrum(K, None, re, im, method="dense")
    assert np.max(np.abs(ps_lu.resolvent * dist - 1)) < 1e-10
    assert np.max(np.abs(ps_dense.resolvent * dist - 1)) < 1e-10
    ps_proj = an.pseudospectrum(K, None, re, im, method="projected", V=X[:, :8])
    assert np.all(ps_proj.resolvent <= ps_lu.resolvent * (1 + 1e-8))
    dist8 = np.min(np.abs(Z[..., None] - lam[None, None, :8]), axis=-1)
    assert np.max(np.abs(ps_proj.resolvent * dist8 - 1)) < 1e-10
    assert ps_lu.pseudospectral_abscissa(0.1) == -np.inf or np.isfinite(
        ps_lu.pseudospectral_abscissa(0.1))


def test_generalized_hermitian_discs_in_M_norm():
    K = _hermitian(30)
    w = np.random.default_rng(1).uniform(0.5, 2.0, 30)
    M = sp.diags(w).tocsr()
    import scipy.linalg as sla
    lam = sla.eigvalsh(K.toarray(), np.diag(w))
    re, im = _grid(lam[0])
    Z = re[None, :] + 1j * im[:, None]
    dist = np.min(np.abs(Z[..., None] - lam[None, None, :]), axis=-1)
    ps = an.pseudospectrum(K, M, re, im, method="lu")
    assert np.max(np.abs(ps.resolvent * dist - 1)) < 1e-10
    ps2 = an.pseudospectrum(K, M, re, im, method="dense", restrict=True)
    assert np.max(np.abs(ps2.resolvent * dist - 1)) < 1e-10


def _jordan(a=10.0, d=1e-2):
    return sp.csr_matrix(np.array([[1.0, a], [0.0, 1.0 + d]], dtype=complex)), a, d


def test_jordan_block_resolvent_and_henrici():
    J, a, d = _jordan()
    z = 1.0 + 0.5 * d + 0.3j
    R = np.linalg.inv(J.toarray() - z * np.eye(2))
    assert abs(an.resolvent_norm(J, None, z) - np.linalg.norm(R, 2)) < 1e-9 * np.linalg.norm(R, 2)
    line = an.resolvent_along_line(J, None, [z, z + 1j], method="dense", sigma=0.4 + 0.2j)
    assert abs(line[0] - np.linalg.norm(R, 2)) < 1e-8 * np.linalg.norm(R, 2)
    red = an.reduced_operator(J, None, sigma=0.5 + 0.1j)
    h = an.henrici(red.A)
    assert abs(h["departure"] - a) < 1e-8
    c = an.mode_nonnormality(J, None, [1.0, 1.0 + d], reduced=red, fraction=1e-3)
    kappa = np.sqrt(1 + (a / d) ** 2)
    assert np.all(c > 0.9 * kappa) and np.all(c < 1.01 * kappa)


def test_mode_nonnormality_one_for_normal():
    K = _hermitian(20)
    lam = np.linalg.eigvalsh(K.toarray())
    c = an.mode_nonnormality(K, None, lam[:4], lam_all=lam)
    assert np.allclose(c, 1.0, atol=1e-8)


def test_numerical_abscissa_and_range():
    """For dt x = -A x with A = [[1, a], [0, 1]] the numerical abscissa is
    -1 + a/2, the spectral abscissa -1, and the numerical range the disc of
    radius a/2 about 1."""
    a = 3.0
    A = np.array([[1.0, a], [0.0, 1.0]], dtype=complex)
    na = an.numerical_abscissa(A)
    assert abs(na["numerical"] - (-1 + a / 2)) < 1e-12
    assert abs(na["spectral"] + 1) < 1e-12
    pts = an.numerical_range(A, 64)
    assert np.max(np.abs(np.abs(pts - 1.0) - a / 2)) < 1e-10
    Ah = an.reduced_operator(_hermitian(12), None, sigma=0.2j).A
    assert abs(an.numerical_abscissa(Ah)["excess"]) < 1e-10


def test_kreiss_constant_normal_damped():
    """Damped normal modes lambda_j = g_j + i w_j (Re > 0 damped): the Kreiss
    constant sup_{Re z < 0} (-Re z)/dist(z, spectrum) is 1, approached as
    -Re z -> infinity; on a finite grid the bound is below 1."""
    lam = np.array([0.5 + 1j, 1.0 - 0.5j, 2.0 + 0.2j])
    K = sp.diags(lam).tocsr()
    re = np.linspace(-3, 0.5, 15) + 0.013
    im = np.linspace(-2, 2, 9) + 0.017
    ps = an.pseudospectrum(K, None, re, im, method="lu")
    kc = an.kreiss_constant(ps)
    assert 0.5 < kc <= 1.0 + 1e-12
    J, a, d = _jordan()
    psj = an.pseudospectrum(J, None, np.linspace(-1, 0.5, 7), np.linspace(-1, 1, 7), method="lu")
    assert an.kreiss_constant(psj) > 1.0


def test_plot_writes_file(tmp_path):
    K = _hermitian(10)
    lam = np.linalg.eigvalsh(K.toarray())
    ps = an.pseudospectrum(K, None, np.linspace(lam[0] - 1, lam[0] + 1, 5) + 0.01,
                           np.linspace(-1, 1, 5) + 0.01, method="dense")
    out = ps.plot(str(tmp_path / "ps.png"), levels=(1e-1, 1e-2))
    assert (tmp_path / "ps.png").stat().st_size > 0 and out.endswith("ps.png")
