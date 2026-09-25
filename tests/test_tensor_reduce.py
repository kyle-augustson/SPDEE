"""Unit tests for spdee.ir.tensor_reduce.

Covers:
- Pure Grad(v) — all 9 spin components match the SymPy-derived formulas.
- Grad(v) + Transpose(Grad(v)) — auto-detected as symmetric (structural).
- as_symmetric on a genuinely symmetric expression succeeds.
- as_symmetric on a non-symmetric expression FAILS loudly.
- Background.Tensor(symmetric=True) accepts symmetric components.
- Background.Tensor(symmetric=True) rejects non-symmetric components.
- Transpose roundtrip (Transpose(Transpose(X)) == X).
- Sum of tensor backgrounds.
- ScalarMultiple of a tensor.
- Cross-check with the strain-BC derivation: E^{±1, 0} formulas.
"""

from __future__ import annotations

import pytest
import sympy as sp

from spdee import (
    Background, Unknown, Grad, Transpose,
    r as r_sym, theta as t_sym,
)
from spdee.ir.tensor_reduce import (
    reduce_tensor_expression, is_structurally_symmetric,
    vector_unknown_spin_placeholders,
    SPIN_INDICES, SPIN_PAIRS,
)


# =============================================================================
# Helpers
# =============================================================================

def _placeholder_dict(field):
    """Return v^- v^0 v^+ SymPy placeholders for an unknown vector field."""
    return vector_unknown_spin_placeholders(field)


# =============================================================================
# Pure Grad(v) — all 9 spin components
# =============================================================================

class TestGradSpinComponents:
    """Verify the 9-component Grad(v) formulas match the derivation."""

    def test_grad_v_at_m0_diagonal(self):
        v = Unknown.Vector("v_test")
        red = reduce_tensor_expression(Grad(v), m=0)
        ph = _placeholder_dict(v)
        # (0, 0) = d_r v^0
        assert sp.simplify(red[(0, 0)] - sp.diff(ph[0], r_sym)) == 0
        # (0, -1) = d_r v^+   (V19 inverse-mapping convention)
        assert sp.simplify(red[(0, -1)] - sp.diff(ph[+1], r_sym)) == 0
        # (0, +1) = d_r v^-
        assert sp.simplify(red[(0, +1)] - sp.diff(ph[-1], r_sym)) == 0

    def test_grad_v_off_diagonal_at_m0(self):
        v = Unknown.Vector("v_test")
        red = reduce_tensor_expression(Grad(v), m=0)
        ph = _placeholder_dict(v)
        sqrt2 = sp.sqrt(2)
        # (+1, 0)|m=0 = -v^-/r + (1/(sqrt(2) r)) d_theta v^0
        expected = (-ph[-1] / r_sym
                    + sp.diff(ph[0], t_sym) / (sqrt2 * r_sym))
        assert sp.simplify(red[(+1, 0)] - expected) == 0
        # (-1, 0)|m=0 = -v^+/r + (1/(sqrt(2) r)) d_theta v^0
        expected_m = (-ph[+1] / r_sym
                      + sp.diff(ph[0], t_sym) / (sqrt2 * r_sym))
        assert sp.simplify(red[(-1, 0)] - expected_m) == 0

    def test_grad_v_at_m2_has_m_dependence(self):
        v = Unknown.Vector("v_test")
        red = reduce_tensor_expression(Grad(v), m=2)
        ph = _placeholder_dict(v)
        sqrt2 = sp.sqrt(2)
        # (+1, 0)|m = (1/(sqrt(2) r)) d_theta v^0 + (m/(sqrt(2) r sin theta)) v^0 - v^-/r
        expected = (-ph[-1] / r_sym
                    + (sp.diff(ph[0], t_sym)
                       + 2 / sp.sin(t_sym) * ph[0]) / (sqrt2 * r_sym))
        assert sp.simplify(red[(+1, 0)] - expected) == 0

    def test_grad_v_components_at_negative_m(self):
        """The m=+2 and m=-2 cases should differ only in the m-dependent term."""
        v = Unknown.Vector("v_test")
        red_p = reduce_tensor_expression(Grad(v), m=+2)
        red_n = reduce_tensor_expression(Grad(v), m=-2)
        # The "m" coefficient flips sign. Specific components that should flip:
        # On (+1, 0), the m-term is + ; on (-1, 0), it's -.
        # diff = red_p - red_n should be twice the m-dependent piece.
        ph = _placeholder_dict(v)
        sqrt2 = sp.sqrt(2)
        # m-coefficient on (+1, 0) is m * v^0 / (sqrt(2) r sin theta)
        diff_p0 = sp.simplify(red_p[(+1, 0)] - red_n[(+1, 0)])
        expected_diff = 4 * ph[0] / (sqrt2 * r_sym * sp.sin(t_sym))  # 2 - (-2) = 4
        assert sp.simplify(diff_p0 - expected_diff) == 0


# =============================================================================
# Grad(v) is NOT structurally symmetric
# =============================================================================

class TestStructuralSymmetryDetection:
    def test_grad_alone_is_not_symmetric(self):
        v = Unknown.Vector("v_test")
        expr = Grad(v)
        assert not is_structurally_symmetric(expr)
        red = reduce_tensor_expression(expr, m=0)
        assert not red.symmetric

    def test_grad_plus_transpose_grad_is_symmetric(self):
        v = Unknown.Vector("v_test")
        expr = Grad(v) + Transpose(Grad(v))
        assert is_structurally_symmetric(expr)
        red = reduce_tensor_expression(expr, m=0)
        assert red.symmetric

    def test_transpose_grad_plus_grad_is_symmetric(self):
        """Reverse argument order should also be detected."""
        v = Unknown.Vector("v_test")
        expr = Transpose(Grad(v)) + Grad(v)
        assert is_structurally_symmetric(expr)
        red = reduce_tensor_expression(expr, m=0)
        assert red.symmetric

    def test_transpose_of_symmetric_is_symmetric(self):
        v = Unknown.Vector("v_test")
        sym = Grad(v) + Transpose(Grad(v))
        # Transpose of a symmetric expression.
        expr = Transpose(sym)
        assert is_structurally_symmetric(expr)
        red = reduce_tensor_expression(expr, m=0)
        assert red.symmetric

    def test_scalar_multiple_of_symmetric_is_symmetric(self):
        v = Unknown.Vector("v_test")
        sym = Grad(v) + Transpose(Grad(v))
        expr = 3.0 * sym
        assert is_structurally_symmetric(expr)
        red = reduce_tensor_expression(expr, m=0)
        assert red.symmetric

    def test_negation_of_symmetric_is_symmetric(self):
        v = Unknown.Vector("v_test")
        sym = Grad(v) + Transpose(Grad(v))
        expr = -sym
        assert is_structurally_symmetric(expr)
        red = reduce_tensor_expression(expr, m=0)
        assert red.symmetric

    def test_sum_sym_plus_general_is_not_symmetric(self):
        """sym + general is NOT auto-detected as symmetric."""
        v = Unknown.Vector("v_test")
        w = Unknown.Vector("w_test")
        sym = Grad(v) + Transpose(Grad(v))
        gen = Grad(w)  # non-symmetric
        expr = sym + gen
        assert not is_structurally_symmetric(expr)

    def test_two_separately_constructed_grads_are_structurally_equal(self):
        """Grad(v) + Transpose(Grad(v)) with separately-constructed Grads
        is still detected as symmetric (uses structural equality, not is)."""
        v = Unknown.Vector("v_test")
        g1 = Grad(v)
        g2 = Grad(v)  # separate Python object
        assert g1 is not g2  # confirm separateness
        expr = g1 + Transpose(g2)
        assert is_structurally_symmetric(expr)


# =============================================================================
# as_symmetric() override
# =============================================================================

class TestAsSymmetricOverride:
    def test_as_symmetric_on_strain_succeeds(self):
        """as_symmetric on Grad(v) + Transpose(Grad(v)) — verification passes."""
        v = Unknown.Vector("v_test")
        expr = (Grad(v) + Transpose(Grad(v))).as_symmetric()
        red = reduce_tensor_expression(expr, m=0)
        assert red.symmetric

    def test_as_symmetric_on_grad_alone_fails(self):
        """as_symmetric on Grad(v) — verification fails loudly."""
        v = Unknown.Vector("v_test")
        with pytest.raises(ValueError, match="as_symmetric.*FAILED"):
            reduce_tensor_expression(Grad(v).as_symmetric(), m=0)

    def test_as_symmetric_on_nonsymmetric_sum_fails(self):
        """Grad(v) + Grad(w) is not symmetric in general."""
        v = Unknown.Vector("v_test")
        w = Unknown.Vector("w_test")
        expr = (Grad(v) + Grad(w)).as_symmetric()
        with pytest.raises(ValueError, match="as_symmetric.*FAILED"):
            reduce_tensor_expression(expr, m=0)

    def test_as_symmetric_on_scalar_raises_typeerror(self):
        """Cannot call as_symmetric() on a non-tensor expression."""
        v = Unknown.Vector("v_test")
        from spdee import Div
        scalar_expr = Div(v)  # scalar kind
        with pytest.raises(TypeError, match="tensor expression"):
            scalar_expr.as_symmetric()


# =============================================================================
# Background.Tensor symmetric flag
# =============================================================================

class TestBackgroundTensorSymmetricFlag:
    def test_symmetric_true_accepts_symmetric_components(self):
        """A symmetric tensor field with symmetric components is accepted."""
        T = Background.Tensor(
            "T_sym",
            components={
                ("r", "r"): r_sym ** 2,
                ("theta", "theta"): r_sym * sp.cos(t_sym),
                ("phi", "phi"): sp.S(1),
                ("r", "theta"): r_sym * sp.sin(t_sym),
                ("theta", "r"): r_sym * sp.sin(t_sym),  # match
                ("r", "phi"): t_sym ** 2,
                ("phi", "r"): t_sym ** 2,  # match
                ("theta", "phi"): sp.S(0),
                ("phi", "theta"): sp.S(0),  # match
            },
            symmetric=True,
        )
        assert T.symmetric is True

    def test_symmetric_true_rejects_nonsymmetric_components(self):
        """A non-symmetric tensor field declared symmetric=True is rejected."""
        with pytest.raises(ValueError, match="symmetric=True"):
            Background.Tensor(
                "T_bad",
                components={
                    ("r", "theta"): r_sym * sp.sin(t_sym),
                    ("theta", "r"): r_sym * sp.cos(t_sym),  # different!
                },
                symmetric=True,
            )

    def test_symmetric_false_default(self):
        """Default symmetric flag is False; non-symmetric is fine."""
        T = Background.Tensor(
            "T_asym",
            components={
                ("r", "theta"): sp.sin(t_sym),
                ("theta", "r"): sp.cos(t_sym),  # different
            },
        )
        assert T.symmetric is False

    def test_symmetric_background_propagates_through_reducer(self):
        """A symmetric background tensor is reduced as symmetric."""
        T = Background.Tensor(
            "T_sym",
            components={
                ("r", "r"): r_sym,
                ("theta", "r"): sp.cos(t_sym),
                ("r", "theta"): sp.cos(t_sym),  # match
            },
            symmetric=True,
        )
        from spdee.ir.expressions import FieldRef
        expr = FieldRef(T)
        red = reduce_tensor_expression(expr, m=0)
        assert red.symmetric


# =============================================================================
# Transpose roundtrip
# =============================================================================

class TestTransposeRoundtrip:
    def test_transpose_twice_is_identity_on_components(self):
        v = Unknown.Vector("v_test")
        expr1 = Grad(v)
        expr2 = Transpose(Transpose(expr1))
        red1 = reduce_tensor_expression(expr1, m=0)
        red2 = reduce_tensor_expression(expr2, m=0)
        for pair in SPIN_PAIRS:
            assert sp.simplify(red1[pair] - red2[pair]) == 0

    def test_transpose_swaps_spin_pair_indices(self):
        v = Unknown.Vector("v_test")
        red = reduce_tensor_expression(Grad(v), m=0)
        red_T = reduce_tensor_expression(Transpose(Grad(v)), m=0)
        # (sigma_1, sigma_2) of Transpose should equal (sigma_2, sigma_1) of original
        for s1 in SPIN_INDICES:
            for s2 in SPIN_INDICES:
                diff = sp.simplify(red[(s2, s1)] - red_T[(s1, s2)])
                assert diff == 0, (
                    f"Transpose mismatch at ({s1:+d}, {s2:+d}): "
                    f"red[({s2:+d},{s1:+d})] - red_T[({s1:+d},{s2:+d})] = {diff}"
                )


# =============================================================================
# Sum of tensor backgrounds
# =============================================================================

class TestSumOfBackgroundTensors:
    def test_sum_components_add(self):
        T1 = Background.Tensor(
            "T1", components={("r", "r"): r_sym, ("theta", "theta"): r_sym ** 2}
        )
        T2 = Background.Tensor(
            "T2", components={("r", "r"): sp.S(2), ("theta", "theta"): r_sym}
        )
        from spdee.ir.expressions import FieldRef
        expr = FieldRef(T1) + FieldRef(T2)
        red = reduce_tensor_expression(expr, m=0)
        # T1 has T1^{0,0} = T1_rr = r; T2 has T2^{0,0} = T2_rr = 2.
        # Sum: 0,0 component = r + 2.
        assert sp.simplify(red[(0, 0)] - (r_sym + 2)) == 0


# =============================================================================
# ScalarMultiple of tensor
# =============================================================================

class TestScalarMultipleOfTensor:
    def test_scalar_times_grad(self):
        v = Unknown.Vector("v_test")
        expr = 3.5 * Grad(v)
        red = reduce_tensor_expression(expr, m=0)
        ph = _placeholder_dict(v)
        # (0, 0) = 3.5 * d_r v^0
        assert sp.simplify(red[(0, 0)] - sp.Rational(7, 2) * sp.diff(ph[0], r_sym)) == 0


# =============================================================================
# Cross-check: strain-BC formulas
# =============================================================================

class TestStrainBCCrossCheck:
    """Verify the reducer reproduces the strain-BC E^{±1, 0} formulas."""

    def test_E_plus_1_zero_at_m2(self):
        """E^{+1, 0} = ∂_r v^- - v^-/r + (1/(sqrt(2) r))(∂_θ v^0 + m/sin θ · v^0)"""
        v = Unknown.Vector("v_test")
        m = 2
        red = reduce_tensor_expression(
            Grad(v) + Transpose(Grad(v)),
            m=m,
        )
        ph = _placeholder_dict(v)
        sqrt2 = sp.sqrt(2)
        expected = (
            sp.diff(ph[-1], r_sym) - ph[-1] / r_sym
            + (sp.diff(ph[0], t_sym) + m / sp.sin(t_sym) * ph[0]) / (sqrt2 * r_sym)
        )
        diff = sp.simplify(red[(+1, 0)] - expected)
        assert diff == 0, f"E^(+1,0) mismatch: {diff}"

    def test_E_minus_1_zero_at_m_minus_3(self):
        """E^{-1, 0} formula at m=-3."""
        v = Unknown.Vector("v_test")
        m = -3
        red = reduce_tensor_expression(
            Grad(v) + Transpose(Grad(v)),
            m=m,
        )
        ph = _placeholder_dict(v)
        sqrt2 = sp.sqrt(2)
        expected = (
            sp.diff(ph[+1], r_sym) - ph[+1] / r_sym
            + (sp.diff(ph[0], t_sym) - m / sp.sin(t_sym) * ph[0]) / (sqrt2 * r_sym)
        )
        diff = sp.simplify(red[(-1, 0)] - expected)
        assert diff == 0, f"E^(-1,0) mismatch at m=-3: {diff}"


# =============================================================================
# Type errors
# =============================================================================

class TestTypeErrors:
    def test_reduce_non_tensor_raises(self):
        v = Unknown.Vector("v_test")
        from spdee.ir.expressions import FieldRef
        with pytest.raises(TypeError, match="tensor expression"):
            reduce_tensor_expression(FieldRef(v), m=0)

    def test_transpose_of_vector_raises(self):
        v = Unknown.Vector("v_test")
        with pytest.raises(TypeError, match="Transpose requires a tensor"):
            Transpose(v)

    def test_reduce_unknown_tensor_raises_not_implemented(self):
        """Tensor unknowns are not yet supported as primary unknowns
        in tensor expressions."""
        T = Unknown.Tensor("T_unk")
        from spdee.ir.expressions import FieldRef
        with pytest.raises(NotImplementedError, match="Tensor unknowns"):
            reduce_tensor_expression(FieldRef(T), m=0)


# =============================================================================
# Backgrounds against the symbolic vector calculus
# =============================================================================

_I, _S2 = sp.I, sp.sqrt(2)
# covariant basis vectors e_sigma, physical order (r, theta, phi)
_E_COV = {+1: (0, 1 / _S2, -_I / _S2), 0: (1, 0, 0), -1: (0, 1 / _S2, _I / _S2)}
_PTS = [(0.7, 0.9), (1.3, 2.1)]


def covariant_spin(T):
    """T_{sigma_1 sigma_2} = (e_sigma_1)_i (e_sigma_2)_j T_ij of a vc.Ten."""
    return {(a, b): sum(_E_COV[a][i] * _E_COV[b][j] * T[i][j]
                        for i in range(3) for j in range(3)) for (a, b) in SPIN_PAIRS}


def contravariant_spin(v):
    """v^sigma of a vc.Vec."""
    return {+1: (v[1] + _I * v[2]) / _S2, 0: v[0], -1: (v[1] - _I * v[2]) / _S2}


def substitute_trial(expr, placeholders, trial_spin):
    """The expression with the placeholders replaced by a trial vector."""
    for sigma, fn in placeholders.items():
        expr = expr.replace(fn.func, sp.Lambda((r_sym, t_sym), trial_spin[sigma]))
    return expr.doit()


def assert_pointwise_equal(got, want, label=""):
    f = sp.lambdify((r_sym, t_sym), got - want, modules="numpy")
    for rv, tv in _PTS:
        d = complex(f(rv, tv))
        assert abs(d) < 1e-10, (label, rv, tv, d)


def analytic_backgrounds():
    """A background tensor and vector with phi components, as IR fields
    and as vector-calculus objects."""
    from analytic import vector_calculus as vc
    z, s = r_sym * sp.cos(t_sym), sp.sin(t_sym)
    c0 = 1 + r_sym ** 2 / 3 + z / 5
    comps = {("r", "r"): c0 + z, ("r", "theta"): -r_sym * s, ("r", "phi"): r_sym ** 2 * s,
             ("theta", "phi"): sp.cos(t_sym) * r_sym, ("phi", "theta"): r_sym ** 3,
             ("phi", "phi"): c0, ("phi", "r"): 2 * r_sym}
    names = ("r", "theta", "phi")
    T0 = Background.Tensor("T0", comps)
    T0v = vc.Ten([[comps.get((names[i], names[j]), 0) for j in range(3)] for i in range(3)])
    B0 = Background.Vector("B0", v_r=sp.cos(t_sym) * (1 + r_sym ** 2 / 4) + r_sym / 4,
                           v_theta=-sp.sin(t_sym) * (1 + r_sym ** 2 / 4), v_phi=r_sym * s / 2)
    B0v = vc.Vec(B0.v_r, B0.v_theta, B0.v_phi)
    trial = vc.Vec(r_sym ** 2 * sp.cos(t_sym), r_sym * sp.sin(t_sym) ** 2,
                   r_sym ** 3 * sp.sin(t_sym) * sp.cos(t_sym))
    return T0, T0v, B0, B0v, trial


@pytest.mark.parametrize("m", [0, 1, 2])
def test_background_tensor_leaf_is_covariant(m):
    """T0 + Grad(v) reduces to the covariant components of the physical
    sum: the background leaf is lowered from the contravariant
    spin_components() (sigma -> -sigma on both indices)."""
    from analytic import vector_calculus as vc
    T0, T0v, _B0, _B0v, trial = analytic_backgrounds()
    v = Unknown.Vector("v")
    red = reduce_tensor_expression(T0 + Grad(v), m=m)
    want = covariant_spin(vc.grad(trial, m) + T0v)
    ph = _placeholder_dict(v)
    for key in SPIN_PAIRS:
        assert_pointwise_equal(substitute_trial(red[key], ph, contravariant_spin(trial)),
                               want[key], key)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_grad_of_background_vector_is_axisymmetric(m):
    """Grad(B0 + v): the background carries no e^{i m phi}, so its
    gradient is taken at m = 0 while the unknown's is taken at m."""
    from analytic import vector_calculus as vc
    _T0, _T0v, B0, B0v, trial = analytic_backgrounds()
    v = Unknown.Vector("v")
    red = reduce_tensor_expression(Grad(B0 + v), m=m)
    want = covariant_spin(vc.grad(trial, m) + vc.grad(B0v, 0))
    ph = _placeholder_dict(v)
    for key in SPIN_PAIRS:
        assert_pointwise_equal(substitute_trial(red[key], ph, contravariant_spin(trial)),
                               want[key], key)
    if m:
        wrong = covariant_spin(vc.grad(trial + B0v, m))
        assert any(abs(sp.lambdify((r_sym, t_sym), want[k] - wrong[k])(0.7, 0.9)) > 1e-3
                   for k in SPIN_PAIRS)
