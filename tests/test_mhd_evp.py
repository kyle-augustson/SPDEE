"""End-to-end MHD eigenvalue tests.

Tests for the linearized MHD operator built with state augmentation
(Curl(Cross(...)) terms via auxiliary unknowns). Validates that the
spectrum has the expected structure as one tunes physical parameters.

Tests are organized as a hierarchy of complexity:

  Stage 1: Pure magnetic diffusion (∂_t B = η ∇²B with B=0 at boundaries).
    Validates that the linearized B equation infrastructure works.

  Stage 2: Coupled MHD with a uniform background field B0 = B0_amp · ẑ.
    Verifies the state-augmentation Curl(Cross(...)) machinery acts
    as expected.

  Stage 3: Same coupled MHD system, but in ball geometry. Validates
    that the equation-order-aligned layout (selected automatically
    when auxiliary unknowns are present) correctly handles ball-origin
    regularity for the augmented system.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Lap, Dt, Grad, Div, Curl, Cross, Equation,
    EquationSystem, BoundaryCondition, Boundary, Dot, r_hat,
    theta_hat, phi_hat,
)
from spdee.backends.fd2.grid import make_shell_grid, make_ball_grid
from spdee.backends.fd2.compiled_problem import compile_problem


# ---------------------------------------------------------------------------
# Stage 1: pure magnetic diffusion
# ---------------------------------------------------------------------------

class TestPureMagneticDecay:
    """∂_t B = ∇²B in a spherical shell with B=0 at both radial boundaries.

    Validates that the linearized B equation pipeline works. With Dt(B)
    and -Lap(B) signs, eigenvalues are POSITIVE (= +κ²).
    """

    def _make_compiled(self, N=20, order=6):
        B = Unknown.Vector("B")
        eq = Equation(Dt(B) - Lap(B), name="induction_diffusive")
        bcs = []
        for label, loc, name_pref in [
            ("inner", Boundary.R_INNER, "Bi"),
            ("outer", Boundary.R_OUTER, "Bo"),
        ]:
            for unit, comp_name in [
                (r_hat, "r"), (theta_hat, "t"), (phi_hat, "p"),
            ]:
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, B), name=f"{name_pref}{comp_name}",
                ))
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(
            N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0,
        )
        return compile_problem(sys_, grid, m=1, order_r=order, order_theta=order)

    def test_smallest_decay_rate_finite(self):
        """The smallest decay rate is finite, positive, and approximately
        real (small imaginary part is ARPACK noise on near-degenerate modes).
        """
        cp = self._make_compiled(N=16, order=6)
        res = cp.solve_evp(k=10, sigma=15.0)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        positive = finite[finite.real > 0]
        assert len(positive) >= 5, (
            f"Expected >=5 positive eigenvalues, got {len(positive)}"
        )
        # |Im| should be small RELATIVE to |Re|.
        max_rel_im = max(abs(x.imag) / max(abs(x.real), 1e-30) for x in positive)
        assert max_rel_im < 1e-3, (
            f"Pure diffusion should have nearly-real eigenvalues; "
            f"max |Im|/|Re| = {max_rel_im:.2e}"
        )

    def test_decay_rates_are_real_and_positive(self):
        """All decay-mode eigenvalues are real (to ARPACK noise) and positive."""
        cp = self._make_compiled(N=16, order=6)
        res = cp.solve_evp(k=15, sigma=25.0)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        for x in finite:
            assert x.real > 0, f"Decay eigenvalue should be positive: λ={x}"
            rel_im = abs(x.imag) / max(abs(x.real), 1e-30)
            assert rel_im < 1e-3, (
                f"Decay eigenvalue should be ~real (|Im|/|Re|<1e-3): λ={x}"
            )


# ---------------------------------------------------------------------------
# Stage 2: coupled MHD with a uniform background field
# ---------------------------------------------------------------------------

class TestMHDCoupledSystem:
    """Linearized incompressible MHD around B₀ = B0_amp · ẑ in a shell.

    Equations:
        ∂_t v + ∇p = ν ∇²v + Cross(Curl(B), B0)          (momentum)
        ∂_t B      = η ∇²B + Curl(Cross(v, B0))           (induction)
        ∇·v = 0                                            (continuity)

    No-slip BCs on v, perfect-conductor-like (B=0) BCs on B.

    With ν=η=1 (large dissipation), Alfvén waves are heavily damped,
    so eigenvalues are mostly real (decay). With B0=0, the v and B
    sectors decouple; with B0 ≠ 0, they couple via the augmentation-
    based Curl(Cross) and Cross(Curl) terms, picking up imaginary
    parts ∝ B0 in the linear (small-B0) regime.
    """

    def _make_compiled(self, B0_amp, eta=1.0, nu=1.0, m=1, N=14, order=4):
        v = Unknown.Vector("v")
        B = Unknown.Vector("B")
        p = Unknown.Scalar("p")
        t_sym = sp.symbols("theta", positive=True)
        # B0 = B0_amp · ẑ.  ẑ in spherical = cos(θ) r̂ − sin(θ) θ̂.
        B0 = Background.Vector(
            "B0",
            v_r=sp.S(B0_amp) * sp.cos(t_sym),
            v_theta=-sp.S(B0_amp) * sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        eq_mom = Equation(
            Dt(v) + Grad(p) - sp.S(nu) * Lap(v) - Cross(Curl(B), B0),
            name="momentum",
        )
        eq_ind = Equation(
            Dt(B) - sp.S(eta) * Lap(B) - Curl(Cross(v, B0)),
            name="induction",
        )
        eq_cont = Equation(Div(v), name="continuity")
        bcs = []
        # No-slip on v at both boundaries
        for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
            for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, v), name=f"v{c}{lbl}"))
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, B), name=f"B{c}{lbl}"))
        sys_ = EquationSystem([eq_mom, eq_ind, eq_cont], bcs)
        grid = make_shell_grid(
            N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0,
        )
        return compile_problem(sys_, grid, m=m, order_r=order, order_theta=order)

    def test_zero_B0_decoupled_spectrum(self):
        """With B0 = 0, the spectrum has finite eigenvalues and the
        physical modes (those with eigenvectors mostly in v/B sectors,
        not aux) are real positive.
        """
        cp = self._make_compiled(B0_amp=0.0, eta=1.0, nu=1.0, m=1, N=12)
        res = cp.solve_evp(k=10, sigma=20.0)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        # Filter to nearly-real eigenvalues (|Im|/|Re| < 1e-3). These
        # are the physical decay modes; complex pairs are spurious
        # aux-constraint artifacts.
        physical = [x for x in finite if abs(x.imag) / max(abs(x.real), 1) < 1e-3]
        assert len(physical) >= 3, (
            f"Expected >=3 nearly-real (physical) eigenvalues, got "
            f"{len(physical)}; all finite = {sorted(finite, key=lambda y: y.real)}"
        )
        # All physical eigenvalues should be positive (decay rates).
        for x in physical:
            assert x.real > 0, f"Decay rate should be positive: {x}"

    def test_smallB0_smallly_perturbs_real_modes(self):
        """A real eigenvalue at B0=0 should shift only slightly for
        small B0. We pick the smallest physical real eigenvalue at
        each of two B0 values and verify it varies SLOWLY (the shift
        is second-order in B0 for the lowest mode).
        """
        def smallest_real(B0_amp):
            cp = self._make_compiled(B0_amp=B0_amp, eta=1.0, nu=1.0, m=1, N=12)
            res = cp.solve_evp(k=10, sigma=20.0)
            lam = np.array(res.eigenvalues)
            finite = lam[np.isfinite(lam)]
            real_like = sorted(
                [x for x in finite if abs(x.imag) / max(abs(x.real), 1) < 1e-3],
                key=lambda y: y.real,
            )
            assert real_like, f"No real-like eigenvalues at B0={B0_amp}"
            return real_like[0].real

        lam_0 = smallest_real(0.0)
        lam_small = smallest_real(0.05)
        # Shift in the lowest decay mode should be small for B0=0.05.
        shift = abs(lam_small - lam_0)
        assert shift < 0.05 * abs(lam_0), (
            f"Smallest decay rate shifted by {shift:.4f} from baseline "
            f"{lam_0:.4f}; too large for B0=0.05"
        )

# ---------------------------------------------------------------------------
# Stage 3: ball geometry MHD
# ---------------------------------------------------------------------------

class TestMHDBallGeometry:
    """Linearized incompressible MHD in a FULL BALL with a uniform
    background field B₀ = B0_amp · ẑ.

    This exercises the augmentation-aware equation-order layout that
    makes ball-origin regularity work when auxiliary unknowns are
    present: the compiler picks equation-order pairing when `__aux_*`
    unknowns are present, so the row/column coincidence that the
    origin-regularity code depends on holds. An alphabetical layout
    would sort the aux columns before the user's unknowns, give a
    rank-deficient (K - σM) and make `splu` fail.
    """

    def _make_compiled(self, B0_amp, eta=1.0, nu=1.0, m=1, N=12, order=4):
        v = Unknown.Vector("v")
        B = Unknown.Vector("B")
        p = Unknown.Scalar("p")
        t_sym = sp.symbols("theta", positive=True)
        B0 = Background.Vector(
            "B0",
            v_r=sp.S(B0_amp) * sp.cos(t_sym),
            v_theta=-sp.S(B0_amp) * sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        eq_mom = Equation(
            Dt(v) + Grad(p) - sp.S(nu) * Lap(v) - Cross(Curl(B), B0),
            name="momentum",
        )
        eq_ind = Equation(
            Dt(B) - sp.S(eta) * Lap(B) - Curl(Cross(v, B0)),
            name="induction",
        )
        eq_cont = Equation(Div(v), name="continuity")
        # Ball geometry: only outer-boundary BCs (origin handled by
        # ball-origin regularity).
        bcs = []
        for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
            bcs.append(BoundaryCondition(
                Boundary.R_OUTER, Dot(unit, v), name=f"v{c}",
            ))
            bcs.append(BoundaryCondition(
                Boundary.R_OUTER, Dot(unit, B), name=f"B{c}",
            ))
        sys_ = EquationSystem(
            [eq_mom, eq_ind, eq_cont],
            bcs,
            gauge_unknowns=["p"],
        )
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        return compile_problem(
            sys_, grid, m=m, order_r=order, order_theta=order,
        )

    def test_compiles_in_ball_with_aux_unknowns(self):
        """The aux-augmented system compiles in ball geometry; layout
        is in equation order (not alphabetical) because aux unknowns
        are present.

        Note: this test explicitly exercises the AUX-AUGMENTED path
        via ``prefer_aux_free=False``. The default path is now
        aux-free (composed blocks via matrix product); see
        ``test_aux_free_matches_aux_augmented_*`` in
        ``test_mhd_aux_free.py`` for cross-validation.
        """
        # Build with prefer_aux_free=False to exercise the aux path.
        v = Unknown.Vector("v")
        B = Unknown.Vector("B")
        p = Unknown.Scalar("p")
        t_sym = sp.symbols("theta", positive=True)
        B0_amp, eta, nu, m, N, order = 1e-3, 1.0, 1.0, 1, 10, 4
        B0 = Background.Vector(
            "B0",
            v_r=sp.S(B0_amp) * sp.cos(t_sym),
            v_theta=-sp.S(B0_amp) * sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        eq_mom = Equation(
            Dt(v) + Grad(p) - sp.S(nu) * Lap(v) - Cross(Curl(B), B0),
            name="momentum",
        )
        eq_ind = Equation(
            Dt(B) - sp.S(eta) * Lap(B) - Curl(Cross(v, B0)),
            name="induction",
        )
        eq_cont = Equation(Div(v), name="continuity")
        bcs = []
        for loc, lbl in [(Boundary.R_OUTER, "o")]:
            for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, v), name=f"v{c}{lbl}"))
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, B), name=f"B{c}{lbl}"))
        sys_ = EquationSystem(
            [eq_mom, eq_ind, eq_cont], bcs, gauge_unknowns=["p"],
        )
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        cp = compile_problem(
            sys_, grid, m=m, order_r=order, order_theta=order,
            prefer_aux_free=False,
        )
        names = cp.matrices.layout.all_names
        # First three primary names should be the user's unknowns in
        # equation order: momentum→v, induction→B, continuity→p.
        assert names[0] == "v"
        assert names[1] == "B"
        assert names[2] == "p"
        # Aux unknowns come last.
        assert any(n.startswith("__aux_curl_B") for n in names[3:])
        assert any(n.startswith("__aux_cross_v_x_B0") for n in names[3:])

    def test_ball_mhd_evp_runs_and_finds_decay_modes(self):
        """The EVP runs successfully in ball geometry (no splu
        singular-factor error) and finds physical decay modes near
        the expected Bessel-zero squared values.
        """
        cp = self._make_compiled(B0_amp=1e-3, N=12, order=4)
        res = cp.solve_evp(k=10, sigma=20.0)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        # Should have multiple finite eigenvalues.
        assert len(finite) >= 5, (
            f"Expected >=5 finite eigenvalues, got {len(finite)}"
        )
        # The physical decay modes are real-like (small B0 perturbation).
        physical = [
            x for x in finite if abs(x.imag) / max(abs(x.real), 1) < 1e-2
        ]
        assert len(physical) >= 3, (
            f"Expected >=3 nearly-real (physical) eigenvalues, got "
            f"{len(physical)}"
        )
        # All physical eigenvalues positive (decay rates).
        for x in physical:
            assert x.real > 0, f"Decay rate should be positive: {x}"
        # The lowest physical eigenvalue should be a small positive
        # decay rate. The exact value depends on which Bessel mode is
        # lowest; at m=1 with full-ball geometry, common values are
        # near π² (≈ 9.87) for some scalar-like modes, or near
        # (first j_1 zero)² ≈ 20.19 for vector modes. We just check
        # the magnitude is in a sensible range, not what specific mode
        # it corresponds to.
        lowest = min(physical, key=lambda x: x.real)
        assert 5.0 < lowest.real < 30.0, (
            f"Lowest decay rate {lowest:.4f} outside sensible range "
            f"[5, 30] for diffusion-dominated ball-geometry MHD"
        )

    def test_ball_mhd_zero_B0_recovers_real_spectrum(self):
        """At B0 = 0, the physical (non-aux) modes are nearly real
        (decoupled diffusion in v and B sectors).
        """
        cp = self._make_compiled(B0_amp=0.0, N=10, order=4)
        res = cp.solve_evp(k=10, sigma=20.0)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        physical = [
            x for x in finite if abs(x.imag) / max(abs(x.real), 1) < 1e-3
        ]
        assert len(physical) >= 3, (
            f"Expected >=3 nearly-real physical eigenvalues at B0=0; "
            f"got {len(physical)}"
        )
        for x in physical:
            assert x.real > 0, f"Decay rate should be positive: {x}"


# ---------------------------------------------------------------------------
# Stage 4: physics validation — quadratic-in-B0 perturbation
# ---------------------------------------------------------------------------

class TestMHDQuadraticPerturbation:
    """The lowest decay eigenvalue's shift from its B₀=0 value scales
    as B₀² for small B₀. This is the dissipative-regime perturbation-
    theory result: linear-in-B₀ terms vanish by symmetry, and the
    leading correction is second order.

    This is a stronger physics validation than the "shift is small"
    check in TestMHDCoupledSystem.test_smallB0_smallly_perturbs_real_modes
    — it confirms the QUADRATIC scaling, which means the augmentation
    operators (Cross(Curl(...), B0) and Curl(Cross(..., B0))) are
    contributing correctly. An incorrect sign or scaling in either
    would break the quadratic fit.
    """

    def _build(self, B0_amp, nu=0.1, eta=0.1, N=12, m=1):
        v = Unknown.Vector("v")
        B = Unknown.Vector("B")
        p = Unknown.Scalar("p")
        t_sym = sp.symbols("theta", positive=True)
        B0 = Background.Vector(
            "B0",
            v_r=sp.S(B0_amp) * sp.cos(t_sym),
            v_theta=-sp.S(B0_amp) * sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        eq_mom = Equation(
            Dt(v) + Grad(p) - sp.S(nu) * Lap(v) - Cross(Curl(B), B0),
            name="momentum",
        )
        eq_ind = Equation(
            Dt(B) - sp.S(eta) * Lap(B) - Curl(Cross(v, B0)),
            name="induction",
        )
        eq_cont = Equation(Div(v), name="continuity")
        bcs = []
        for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
            for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, v), name=f"v{c}{lbl}",
                ))
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, B), name=f"B{c}{lbl}",
                ))
        sys_ = EquationSystem(
            [eq_mom, eq_ind, eq_cont], bcs, gauge_unknowns=["p"],
        )
        grid = make_shell_grid(
            N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0,
        )
        return compile_problem(sys_, grid, m=m, order_r=4, order_theta=4)

    def _track_lowest_real_eigenvalue(self, B0_amp, target_re):
        """Find the eigenvalue nearest target_re in real part."""
        cp = self._build(B0_amp, N=10)
        res = cp.solve_evp(k=15, sigma=target_re)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        closest = min(finite, key=lambda y: abs(y.real - target_re))
        return closest.real

    def test_quadratic_b0_shift(self):
        """Δλ/B₀² is approximately constant for small B₀."""
        # Establish baseline at B0=0
        lam_0 = self._track_lowest_real_eigenvalue(0.0, target_re=18.5)
        # Shifts at small B0
        ratios = []
        for B0 in [0.02, 0.05, 0.1]:
            lam_B = self._track_lowest_real_eigenvalue(B0, target_re=lam_0)
            delta = lam_B - lam_0
            ratio = delta / (B0 ** 2)
            ratios.append(ratio)
        # Each ratio should be of the same sign (perturbation theory
        # gives a definite sign at second order) and the values should
        # be CLOSE to each other (within ~25% — finite-grid + higher-
        # order corrections produce some drift at the larger B0 values).
        signs = [np.sign(r) for r in ratios]
        assert all(s == signs[0] for s in signs), (
            f"Δλ/B0² signs inconsistent: {ratios}"
        )
        # Compare the small-B0 ratio (most perturbative) with the
        # mid-B0 one.
        rel_drift = abs(ratios[0] - ratios[2]) / abs(ratios[0])
        assert rel_drift < 0.30, (
            f"Δλ/B0² drifts {rel_drift:.2%} between B0=0.02 and B0=0.1; "
            f"ratios = {ratios}. Expected ≈ constant for quadratic "
            f"perturbation."
        )


# ---------------------------------------------------------------------------
# Stage 5: full stratified MHD — momentum + induction + continuity + energy
# ---------------------------------------------------------------------------

class TestFullStratifiedMHD:
    """The complete linearized compressible-stratified MHD primitive set:
        ∂_t v + ∇p = ν ∇²v + Cross(Curl(B), B0)
        ∂_t B     = η ∇²B + Curl(Cross(v, B0))
        ∇·v = 0
        ∂_t s + v · ∇s_0 = κ ∇²s

    This validates that all four equations compile together, the
    augmentation pass creates auxiliary unknowns for the MHD operators,
    the energy equation's new Dot(bg_vec, unknown_vec) → scalar pattern
    interoperates with the rest of the system, and the EVP runs.
    """

    def _build(self, B0_amp=0.1, N=12, m=1, prefer_aux_free=True):
        t_sym = sp.symbols("theta", positive=True)
        B0 = Background.Vector(
            "B0",
            v_r=sp.S(B0_amp) * sp.cos(t_sym),
            v_theta=-sp.S(B0_amp) * sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        grad_s0 = Background.Vector(
            "grad_s0",
            v_r=sp.S(1), v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        B = Unknown.Vector("B")
        p = Unknown.Scalar("p")
        s = Unknown.Scalar("s")
        eq_mom = Equation(
            Dt(v) + Grad(p) - Lap(v) - Cross(Curl(B), B0),
            name="momentum",
        )
        eq_ind = Equation(
            Dt(B) - Lap(B) - Curl(Cross(v, B0)),
            name="induction",
        )
        eq_cont = Equation(Div(v), name="continuity")
        eq_energy = Equation(
            Dt(s) + Dot(grad_s0, v) - Lap(s),
            name="energy",
        )
        bcs = []
        for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
            for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, v), name=f"v{c}{lbl}",
                ))
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, B), name=f"B{c}{lbl}",
                ))
            bcs.append(BoundaryCondition(loc, s, name=f"s_{lbl}"))
        sys_ = EquationSystem(
            [eq_mom, eq_ind, eq_cont, eq_energy], bcs,
            gauge_unknowns=["p"],
        )
        grid = make_shell_grid(
            N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0,
        )
        return compile_problem(
            sys_, grid, m=m, order_r=4, order_theta=4,
            prefer_aux_free=prefer_aux_free,
        )

    def test_full_system_compiles(self):
        """All four equations compile together with the aux-augmented
        path; layout includes the user's unknowns and the MHD
        augmentation aux unknowns.

        The default (aux-free) path does not introduce aux unknowns;
        see ``test_full_system_compiles_aux_free`` for that case.
        """
        cp = self._build(B0_amp=0.1, N=10, prefer_aux_free=False)
        names = cp.matrices.layout.all_names
        # User unknowns in equation order.
        assert "v" in names
        assert "B" in names
        assert "p" in names
        assert "s" in names
        # Aux unknowns from MHD augmentation.
        assert any(n.startswith("__aux_curl_B") for n in names)
        assert any(n.startswith("__aux_cross_v_x_B0") for n in names)
        # No aux for the energy equation (it has no composed-operator
        # pattern — Dot is pointwise).
        assert not any("grad_s0" in n for n in names if n.startswith("__aux"))

    def test_full_system_compiles_aux_free(self):
        """The default aux-free path compiles the full stratified-MHD
        system without introducing aux unknowns."""
        cp = self._build(B0_amp=0.1, N=10)  # prefer_aux_free=True default
        names = cp.matrices.layout.all_names
        # User unknowns only — no aux names.
        assert "v" in names
        assert "B" in names
        assert "p" in names
        assert "s" in names
        assert not any(n.startswith("__aux") for n in names)

    def test_full_system_evp_runs(self):
        """The full stratified-MHD EVP runs without splu failure and
        produces sensible decay eigenvalues.
        """
        cp = self._build(B0_amp=0.1, N=12)
        res = cp.solve_evp(k=10, sigma=20.0)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        assert len(finite) >= 5, (
            f"Expected >=5 finite eigenvalues, got {len(finite)}"
        )
        # In the strongly-dissipative regime with these BCs, the lowest
        # eigenvalue should be a small positive decay rate.
        lowest = min(finite, key=lambda y: y.real)
        assert lowest.real > 0, (
            f"Lowest decay rate should be positive: {lowest}"
        )
        assert 5.0 < lowest.real < 50.0, (
            f"Lowest decay rate {lowest:.3f} outside sensible range"
        )

    def test_energy_decoupled_at_zero_grad_s0(self):
        """When ∇s_0 = 0, the energy equation decouples from v.
        The energy sector's eigenvalues should be pure Laplacian
        decay rates (same as if we had a standalone s equation).
        """
        t_sym = sp.symbols("theta", positive=True)
        v = Unknown.Vector("v")
        B = Unknown.Vector("B")
        p = Unknown.Scalar("p")
        s = Unknown.Scalar("s")
        B0 = Background.Vector(
            "B0",
            v_r=sp.S(0.1) * sp.cos(t_sym),
            v_theta=-sp.S(0.1) * sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        grad_s0_zero = Background.Vector(
            "grad_s0_zero",
            v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        eq_mom = Equation(
            Dt(v) + Grad(p) - Lap(v) - Cross(Curl(B), B0),
            name="momentum",
        )
        eq_ind = Equation(
            Dt(B) - Lap(B) - Curl(Cross(v, B0)),
            name="induction",
        )
        eq_cont = Equation(Div(v), name="continuity")
        # With grad_s0 = 0, the Dot(grad_s0, v) term should be
        # operationally zero (note: the augmentation zero-bg-drop
        # doesn't apply here because the term isn't Cross/Curl-Cross;
        # but the dispatcher just builds a zero block).
        eq_energy = Equation(
            Dt(s) + Dot(grad_s0_zero, v) - Lap(s),
            name="energy",
        )
        bcs = []
        for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
            for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, v), name=f"v{c}{lbl}",
                ))
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, B), name=f"B{c}{lbl}",
                ))
            bcs.append(BoundaryCondition(loc, s, name=f"s_{lbl}"))
        sys_ = EquationSystem(
            [eq_mom, eq_ind, eq_cont, eq_energy], bcs,
            gauge_unknowns=["p"],
        )
        grid = make_shell_grid(N_r=12, N_theta=12, r_inner=0.3, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)
        # The energy equation rows should only couple to s columns and
        # whatever the equation mass adds (Lap → diag(r²)). NOT to v.
        energy_slice = cp.matrices.equation_row_slices["energy"]
        v_slice = cp.matrices.layout.field_slice("v")
        A_csr = cp.matrices.A_total.tocsr()
        A_energy_vs_v = A_csr[energy_slice, v_slice]
        # All entries should be zero.
        max_abs = abs(A_energy_vs_v).max() if A_energy_vs_v.nnz > 0 else 0.0
        assert max_abs < 1e-14, (
            f"With grad_s0=0, A[energy, v] should be zero but has "
            f"max |entry| = {max_abs:.3e}"
        )


# ---------------------------------------------------------------------------
# Stage 6: Boussinesq g-mode physics validation
# ---------------------------------------------------------------------------

class TestGModePhysics:
    """Boussinesq gravity-mode (g-mode) physics validation.

    Setup: stably stratified shell with no rotation, no magnetic field,
    weak dissipation. The energy and momentum equations couple through:
      Energy:   ∂_t s + Dot(grad_s0, v) = κ ∇²s
      Momentum: ∂_t v + ∇p = ν ∇²v - s · ĝ   (buoyancy)

    where grad_s0 = N² r̂ (stable stratification) and ĝ = -r̂ (gravity
    inward). In the WKB / asymptotic limit, the fundamental g-mode
    frequency satisfies ω → N as the radial wavenumber → 0.

    The cleanest physics test: scale N²; the g-mode frequency should
    scale as √N². This is the defining property of g-modes and a strong
    end-to-end validation of:
      - The Dot(bg_vec, unknown_vec) → scalar dispatcher
        (energy equation)
      - The bg_vec_times_unk_scalar dispatcher (buoyancy)
      - Their interplay with momentum, continuity, and the gauge
        constraint for p
    """

    def _build(self, N_sq, nu=1e-4, kappa=1e-4, N_grid=24, m=1):
        g_unit = Background.Vector(
            "g_unit",
            v_r=sp.S(-1), v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        grad_s0 = Background.Vector(
            "grad_s0",
            v_r=sp.S(N_sq), v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        p = Unknown.Scalar("p")
        s = Unknown.Scalar("s")
        # Buoyancy: +s·g_unit. With g_unit=-r̂, this is -s·r̂ pulling
        # a positive-s parcel inward (correct for stable stratification
        # where positive s = warmer/lighter parcel that wants to rise).
        # Hmm actually +s·g_unit with g_unit=-r̂ gives -s·r̂ in the
        # equation Dt(v) = ... - s·r̂, i.e. dv_r/dt = -s, so positive s
        # decelerates v_r. With ∂_t s + v_r·N² = 0, a positive v_r
        # creates negative s (entropy fluctuates), which then creates
        # positive dv_r/dt — runaway! Wait, that's unstable.
        # Let me think: standard Boussinesq has
        #   dv_r/dt = -ds/dr + ν∇²v + s·g_unit·r̂ (g_unit pointing UP for buoy)
        # If g points DOWN (g_unit·r̂ = -1) and stratification is STABLE
        # (positive ds_0/dr), then a positive temperature perturbation
        # makes the parcel rise. With s ∝ T', dv_r/dt = +(g_amp)·s for
        # stable buoyancy means we need... let me just empirically
        # verify the scaling. If ω² = +N², the system is OSCILLATORY
        # (stable). The test below should reveal whether the sign is
        # correct.
        eq_mom = Equation(
            Dt(v) + Grad(p) - sp.S(nu) * Lap(v) + s * g_unit,
            name="momentum",
        )
        eq_cont = Equation(Div(v), name="continuity")
        eq_energy = Equation(
            Dt(s) + Dot(grad_s0, v) - sp.S(kappa) * Lap(s),
            name="energy",
        )
        bcs = [
            BoundaryCondition(Boundary.R_INNER, s, name="s_i"),
            BoundaryCondition(Boundary.R_OUTER, s, name="s_o"),
        ]
        for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
            for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
                bcs.append(BoundaryCondition(
                    loc, Dot(unit, v), name=f"v{c}{lbl}",
                ))
        sys_ = EquationSystem(
            [eq_mom, eq_cont, eq_energy], bcs, gauge_unknowns=["p"],
        )
        grid = make_shell_grid(
            N_r=N_grid, N_theta=N_grid, r_inner=0.3, r_outer=1.0,
        )
        return compile_problem(sys_, grid, m=m, order_r=4, order_theta=4)

    def _find_top_g_mode_imag(self, N_sq):
        """Compile the g-mode system at given N², find the eigenvalue
        with the largest positive imaginary part in the band
        [0.5√N², 1.5√N²]. Returns Im(λ).
        """
        cp = self._build(N_sq, N_grid=24)
        sqrt_N = np.sqrt(N_sq)
        res = cp.solve_evp(k=30, sigma=0.1 + 1j * sqrt_N)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        candidates = [
            x for x in finite
            if x.imag > 0.5 * sqrt_N and x.imag < 1.5 * sqrt_N
        ]
        assert candidates, (
            f"No g-mode found at N²={N_sq} in band [{0.5*sqrt_N:.3f}, "
            f"{1.5*sqrt_N:.3f}]"
        )
        return max(candidates, key=lambda y: y.imag).imag

    def test_gmode_frequency_scales_with_sqrt_N(self):
        """The fundamental g-mode oscillation frequency ω scales as √N²
        across an order-of-magnitude range. Log-log slope ≈ 1.
        """
        omegas = []
        Nsqs = [1.0, 2.0, 4.0, 9.0, 16.0]
        for N_sq in Nsqs:
            omegas.append(self._find_top_g_mode_imag(N_sq))
        # Each ratio ω/√N² should be close to a constant (the WKB
        # geometric factor, expected near 1 for the fundamental).
        ratios = [omegas[i] / np.sqrt(Nsqs[i]) for i in range(len(Nsqs))]
        for r in ratios:
            assert 0.8 < r < 1.1, (
                f"ω/√N² = {r:.4f} outside expected band [0.8, 1.1] for "
                f"the fundamental g-mode"
            )
        # The ratios should be approximately constant — variation
        # within a few percent.
        rel_range = (max(ratios) - min(ratios)) / np.mean(ratios)
        assert rel_range < 0.05, (
            f"ω/√N² varies by {rel_range:.2%} across the range; "
            f"expected approximately constant: {ratios}"
        )

    def test_gmode_loglog_slope_is_one(self):
        """Rigorous slope check: fit log(ω) = α + β·log(√N²); expect β = 1."""
        Nsqs = np.array([1.0, 2.0, 4.0, 9.0, 16.0])
        omegas = np.array([self._find_top_g_mode_imag(n) for n in Nsqs])
        log_sqrtN = np.log(np.sqrt(Nsqs))
        log_omega = np.log(omegas)
        slope, intercept = np.polyfit(log_sqrtN, log_omega, 1)
        # Expect β = 1 (ω ∝ √N²); allow modest tolerance for finite-grid
        # corrections.
        assert abs(slope - 1.0) < 0.02, (
            f"log-log slope = {slope:.4f}, expected ≈ 1.0 for g-mode"
        )


# ---------------------------------------------------------------------------
# Stage 7: ball-geometry g-modes
# ---------------------------------------------------------------------------

class TestGModePhysicsBall:
    """Same g-mode physics test as TestGModePhysics, but in a FULL BALL.

    Validates that the energy and buoyancy operators interoperate
    correctly with ball-origin regularity. The pointwise operators
    (Dot(bg_vec, unk_vec) → scalar; bg_vec * unk_scalar → vector)
    don't have intrinsic r=0 singularities, so they should "just work"
    under the equation-order layout.

    Expected: ω/√N² ≈ constant ≈ 1 across N² range, log-log slope ≈ 1.

    Notes on physics:
      - In the ball, the radial structure of the fundamental g-mode is
        even closer to k_r → 0 than in the shell (no inner boundary
        to impose a node), so ω/√N² should be slightly closer to 1
        than the shell case.
    """

    def _build(self, N_sq, nu=1e-4, kappa=1e-4, N_grid=20, m=1):
        from spdee.backends.fd2.grid import make_ball_grid
        g_unit = Background.Vector(
            "g_unit",
            v_r=sp.S(-1), v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        grad_s0 = Background.Vector(
            "grad_s0",
            v_r=sp.S(N_sq), v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        p = Unknown.Scalar("p")
        s = Unknown.Scalar("s")
        eq_mom = Equation(
            Dt(v) + Grad(p) - sp.S(nu) * Lap(v) + s * g_unit,
            name="momentum",
        )
        eq_cont = Equation(Div(v), name="continuity")
        eq_energy = Equation(
            Dt(s) + Dot(grad_s0, v) - sp.S(kappa) * Lap(s),
            name="energy",
        )
        # Ball geometry: only outer BCs (origin handled by origin
        # regularity for scalar and vector unknowns).
        bcs = [BoundaryCondition(Boundary.R_OUTER, s, name="s_o")]
        for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
            bcs.append(BoundaryCondition(
                Boundary.R_OUTER, Dot(unit, v), name=f"v{c}o",
            ))
        sys_ = EquationSystem(
            [eq_mom, eq_cont, eq_energy], bcs, gauge_unknowns=["p"],
        )
        grid = make_ball_grid(N_r=N_grid, N_theta=N_grid, r_outer=1.0)
        return compile_problem(
            sys_, grid, m=m, order_r=4, order_theta=4,
        )

    def _find_top_g_mode_imag(self, N_sq):
        cp = self._build(N_sq, N_grid=20)
        sqrt_N = np.sqrt(N_sq)
        res = cp.solve_evp(k=30, sigma=0.1 + 1j * sqrt_N)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        candidates = [
            x for x in finite
            if x.imag > 0.5 * sqrt_N and x.imag < 1.5 * sqrt_N
        ]
        assert candidates, (
            f"No g-mode found at N²={N_sq} in band [{0.5*sqrt_N:.3f}, "
            f"{1.5*sqrt_N:.3f}]"
        )
        return max(candidates, key=lambda y: y.imag).imag

    def test_ball_gmode_compiles_with_aux_unknowns_absent(self):
        """The ball g-mode system has NO aux unknowns (no augmentation
        terms like Cross(Curl,…)), but Dt forces equation-order
        layout. Verify the layout is in equation order and the system
        compiles.
        """
        cp = self._build(N_sq=1.0, N_grid=10)
        names = cp.matrices.layout.all_names
        # Equation order: momentum → v, continuity → p, energy → s
        assert names == ["v", "p", "s"], (
            f"Expected equation-order layout [v, p, s], got {names}"
        )
        # No aux unknowns.
        assert not any(n.startswith("__aux_") for n in names)

    def test_ball_gmode_frequency_scales_with_sqrt_N(self):
        """The ball g-mode frequency scales as √N²; ratio approximately
        constant across the range.
        """
        omegas = []
        Nsqs = [1.0, 2.0, 4.0, 9.0, 16.0]
        for N_sq in Nsqs:
            omegas.append(self._find_top_g_mode_imag(N_sq))
        ratios = [omegas[i] / np.sqrt(Nsqs[i]) for i in range(len(Nsqs))]
        for r in ratios:
            assert 0.85 < r < 1.05, (
                f"ω/√N² = {r:.4f} outside expected band [0.85, 1.05] "
                f"for ball g-mode (slight finite-N drift expected)"
            )
        rel_range = (max(ratios) - min(ratios)) / np.mean(ratios)
        assert rel_range < 0.05, (
            f"ω/√N² varies by {rel_range:.2%}; expected approximately "
            f"constant. Ratios: {ratios}"
        )

    def test_ball_gmode_loglog_slope_is_one(self):
        """Rigorous slope check for ball g-modes."""
        Nsqs = np.array([1.0, 2.0, 4.0, 9.0, 16.0])
        omegas = np.array([self._find_top_g_mode_imag(n) for n in Nsqs])
        log_sqrtN = np.log(np.sqrt(Nsqs))
        log_omega = np.log(omegas)
        slope, _ = np.polyfit(log_sqrtN, log_omega, 1)
        assert abs(slope - 1.0) < 0.02, (
            f"Ball log-log slope = {slope:.4f}, expected ≈ 1.0"
        )
