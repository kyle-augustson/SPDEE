"""The backend-agnostic constraint classifier (spdee.ir.constraints).

Every classification outcome on the systems the IR dump freezes and on
the systems below: saddle points (Boussinesq, the weighted anelastic
divergence, the vector potential), involutions written and implied (b
induction), tensor definitions, the Dt-less rule, each refusal; both
``mimetic`` values, and the non-mimetic divergence-cleaning
prescription. ``test_agrees_with_the_staggered_classifier`` calls the
C-grid classifier and this one on every reference system and asserts
the records agree field by field on everything the C-grid one records,
so phase two can make it a thin wrapper.
"""

import pytest
import sympy as sp

from spdee import Background, theta
from spdee.ir import r
from spdee.ir.augment import augment_equation_system
from spdee.ir.backgrounds import SymPyComponent
from spdee.ir.constraints import (
    ClassificationRecord, DivergenceCleaning, INVOLUTION_DROPPED,
    INVOLUTION_SADDLE_POINT, auto_multiplier_name, classify,
)
from spdee.ir.tasks import TASK_GRAD_SCALAR
from spdee.ir.walker import walk_equation_system
from spdee.problem import Problem


# ---------------------------------------------------------------------------
# systems
# ---------------------------------------------------------------------------

def _rotating(pb):
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)


def boussinesq(gauge=False, wall="stress_free"):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("T", "scalar")
    pb.add_unknown("p", "scalar")
    _rotating(pb)
    pb.add_background("grad_T0", v_r=-1)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - 10*T*r_hat - 0.01*lap(v) = 0",
                    name="momentum")
    pb.add_equation("dt(T) + dot(v, grad_T0) - 0.01*lap(T) = 0", name="energy")
    pb.add_equation("div(v) = 0", name="continuity")
    for w in ("inner", "outer"):
        if wall == "stress_free":
            pb.add_stress_free("v", at=w)
        else:
            for u in ("r_hat", "theta_hat", "phi_hat"):
                pb.add_bc(f"dot({u}, v) = 0", at=w)
        pb.add_bc("T = 0", at=w)
    return pb.build(gauge_unknowns=["p"] if gauge else [])


def anelastic():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("S", "scalar")
    pb.add_unknown("p", "scalar")
    pb.add_background("rho0", expr="(1 - r**2/2)**2")
    pb.add_background("er_", v_r=1)
    _rotating(pb)
    pb.add_equation("dt(v) + grad(p) - S*er_ + 2*cross(ez, v) - 0.001*lap(v) = 0",
                    name="momentum")
    pb.add_equation("dt(S) - dot(er_, v) - 0.001*lap(S) = 0", name="energy")
    pb.add_equation("div(rho0*v) = 0", name="continuity")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    pb.add_bc("dot(r_hat, grad(S)) = 0", at="inner")
    pb.add_bc("dot(r_hat, grad(S)) = 0", at="outer")
    return pb.build()


def induction(written=True, diffusion="lap", ball=False, wall="conducting"):
    """Momentum + induction in b with a written or implied Div(b) = 0."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("b", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    diff = {"lap": "- 0.01*lap(b)", "curlcurl": "+ 0.01*curl(curl(b))", "none": ""}[diffusion]
    pb.add_equation(f"dt(b) - curl(cross(v, B0)) {diff} = 0", name="induction")
    if written:
        pb.add_equation("div(b) = 0", name="solenoidal")
    for w in (("outer",) if ball else ("inner", "outer")):
        pb.add_bc("dot(r_hat, v) = 0", at=w)
        pb.add_bc("dot(r_hat, b) = 0", at=w)
        if wall == "conducting":
            pb.add_bc("dot(theta_hat, 0.01*curl(b) - cross(v, B0)) = 0", at=w)
            pb.add_bc("dot(phi_hat, 0.01*curl(b) - cross(v, B0)) = 0", at=w)
    return pb.build()


def vector_potential():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("A", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_unknown("psi", "scalar")
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + grad(p) - 0.01*lap(v) - cross(curl(curl(A)), B0) = 0",
                    name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_equation("dt(A) - cross(v, B0) - 0.01*lap(A) - grad(psi) = 0", name="induction")
    pb.add_equation("div(A) = 0", name="coulomb_gauge")
    for w in ("inner", "outer"):
        pb.add_bc("dot(r_hat, v) = 0", at=w)
        pb.add_bc("dot(theta_hat, A) = 0", at=w)
        pb.add_bc("dot(phi_hat, A) = 0", at=w)
        pb.add_bc("psi = 0", at=w)
    return pb.build(gauge_unknowns=["p", "psi"])


def tensor_definition(expr="grad(v) + trans(grad(v))", dt_T=False):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_unknown("T", "tensor")
    _rotating(pb)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - 0.01*div(T) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_equation(("dt(T) + " if dt_T else "") + f"T - ({expr}) = 0", name="def")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    return pb.build()


def zhang(with_dt=True):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    _rotating(pb)
    pb.add_equation(("dt(v) + " if with_dt else "") + "2*cross(ez, v) + grad(p) = 0",
                    name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="inner")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb.build()


def _momentum(pb, extra=""):
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    _rotating(pb)
    pb.add_equation(f"dt(v) + 2*cross(ez, v) + grad(p){extra} = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="inner")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")


def refusal_relation():
    pb = Problem()
    _momentum(pb)
    pb.add_unknown("T", "scalar")
    pb.add_equation("T - p = 0", name="relation")
    return pb.build()


def refusal_no_grad_no_curl():
    pb = Problem()
    pb.add_unknown("v", "vector")
    _rotating(pb)
    pb.add_equation("dt(v) + 2*cross(ez, v) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    return pb.build()


def refusal_tensor_without_equation():
    pb = Problem()
    pb.add_unknown("S", "tensor")
    _momentum(pb, " + div(S)")
    return pb.build()


def refusal_both_readings():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("b", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_unknown("psi", "scalar")
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_equation("dt(b) - curl(cross(v, B0)) - 0.01*lap(b) + grad(psi) = 0",
                    name="induction")
    pb.add_equation("div(b) = 0", name="solenoidal")
    return pb.build()


def refusal_weighted_involution():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("b", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_background("rho0", expr="1 + r")
    pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_equation("dt(b) - curl(cross(v, B0)) = 0", name="induction")
    pb.add_equation("div(rho0*b) = 0", name="solenoidal")
    return pb.build()


def refusal_non_curl_evolution():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("b", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    _rotating(pb)
    pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_equation("dt(b) - curl(cross(v, B0)) + cross(ez, b) = 0", name="induction")
    pb.add_equation("div(b) = 0", name="solenoidal")
    return pb.build()


def refusal_ambiguous_multiplier():
    pb = Problem()
    pb.add_unknown("q", "scalar")
    _momentum(pb, " + grad(q)")
    return pb.build()


def refusal_two_dt():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("w", "vector")
    pb.add_equation("dt(v) + dt(w) - lap(v) = 0", name="pair")
    pb.add_equation("dt(w) - lap(w) = 0", name="w")
    return pb.build()


def refusal_dt_twice():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_equation("dt(v) - lap(v) = 0", name="one")
    pb.add_equation("dt(v) + v = 0", name="two")
    return pb.build()


def refusal_leftover():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("q", "scalar")
    pb.add_equation("dt(v) - lap(v) + grad(q) = 0", name="momentum")
    return pb.build()


def refusal_definition_mixes_vectors():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("w", "vector")
    pb.add_unknown("T", "tensor")
    pb.add_equation("dt(v) - div(T) = 0", name="v")
    pb.add_equation("dt(w) - lap(w) = 0", name="w")
    pb.add_equation("T - grad(v) - grad(w) = 0", name="def")
    return pb.build()


def refusal_multiplier_name_taken():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("b", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_unknown(auto_multiplier_name("b"), "scalar")
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_equation("dt(b) - curl(cross(v, B0)) = 0", name="induction")
    pb.add_equation(f"dt({auto_multiplier_name('b')}) - lap({auto_multiplier_name('b')}) = 0",
                    name="taken")
    return pb.build()


REFUSALS = {
    "relation": (refusal_relation, "'relation'", "identity_scalar(T) + identity_scalar(p)",
                 "Div(M u) = 0"),
    "no_grad_no_curl": (refusal_no_grad_no_curl, "'continuity'",
                        "found div_vector_scalar(v)", "no Curl evolution"),
    "tensor_without_equation": (refusal_tensor_without_equation, "tensor unknown 'S'",
                                "section 2b", "no tensor equation"),
    "both_readings": (refusal_both_readings, "'solenoidal'", "both readings apply",
                      "'psi' enters 'induction' through Grad alone"),
    "weighted_involution": (refusal_weighted_involution, "'solenoidal'",
                            "Div(M b) = 0 with a weight", "no multiplier"),
    "non_curl_evolution": (refusal_non_curl_evolution, "'solenoidal'",
                           "cross_vector(b)", "not curls of the state"),
    "ambiguous_multiplier": (refusal_ambiguous_multiplier, "'continuity'",
                             "several scalar unknowns", "ambiguous"),
    "two_dt": (refusal_two_dt, "'pair'", "Dt on several unknowns", "['v', 'w']"),
    "dt_twice": (refusal_dt_twice, "'v' is under Dt", "'one'", "'two'"),
    "leftover": (refusal_leftover, "['q']", "neither under Dt", "multiplier"),
    "definition_mixes_vectors": (refusal_definition_mixes_vectors, "'def'",
                                 "mixes the vectors", "['v', 'w']"),
}


def dump_systems():
    """(name, system) of the systems this module adds to the IR dump."""
    yield "boussinesq", boussinesq()
    yield "boussinesq_gauge_noslip", boussinesq(gauge=True, wall="noslip")
    yield "anelastic", anelastic()
    yield "induction_written", induction()
    yield "induction_implied", induction(written=False)
    yield "induction_curlcurl_ball", induction(diffusion="curlcurl", ball=True)
    yield "induction_ideal_radial", induction(diffusion="none", wall="radial")
    yield "vector_potential", vector_potential()
    yield "tensor_definition", tensor_definition()
    yield "tensor_definition_grad", tensor_definition("grad(v)")
    yield "tensor_evolution", tensor_definition(dt_T=True)
    yield "zhang", zhang()
    yield "zhang_dtless", zhang(with_dt=False)
    for name, (build, *_) in REFUSALS.items():
        yield f"refusal_{name}", build()
    yield "refusal_multiplier_name_taken", refusal_multiplier_name_taken()


def reference_systems():
    """(name, system) of the systems the IR dump script freezes, built
    from the same test builders."""
    import test_analytic_evp as tae
    import test_ball_collocated as tbc
    import test_sh_fd_backend as tsh
    import test_staggered_general as tsg
    from analytic import manufactured as mf
    from spdee.applications.solar_eigenmodes import add_magnetic_field, make_polytrope_system
    from spdee.applications.solar_strings import solar_hydro_problem
    from spdee.backgrounds.polytrope import PolytropeBackground

    def system(obj, gauge=()):
        return obj.build(gauge_unknowns=list(gauge)) if hasattr(obj, "build") else obj

    yield "zhang_object", tsg.zhang_system(Ek=0.0)
    yield "zhang_object_dtless", tsg.zhang_system(Ek=0.0, with_dt=False)
    yield "zhang_problem", system(tsg.zhang_problem(Ek=0.0))
    yield "zhang_ball", tsg.zhang_system(Ek=0.0, shell=False, gauge=("p",))
    yield "inertial_tae", system(tae.inertial_problem())
    yield "inertial_tbc", tbc._inertial_system()
    yield "doppler_tae", system(tae.doppler_inertial_problem(0.1))
    for wall in ("stress_free", "noslip"):
        yield f"viscous_object_{wall}", tsg.zhang_system(Ek=1e-3, wall=wall)
        yield f"viscous_problem_{wall}", system(tsg.zhang_problem(Ek=1e-3, wall=wall))
        yield f"viscous_ball_{wall}", tsg.zhang_system(Ek=1e-3, wall=wall, shell=False)
    yield "viscous_shfd_sf", system(tsh.inertial_problem(1e-3))
    yield "solar_hydro", system(solar_hydro_problem())
    yield "solar_hydro_dirichlet", system(solar_hydro_problem(entropy_bc="dirichlet"))
    yield "boussinesq", system(tsg.boussinesq_problem())
    yield "boussinesq_noslip_g", system(tsg.boussinesq_problem(wall="noslip", gravity="r"))
    yield "boussinesq_strain", system(tsg.boussinesq_problem(
        viscous="div(grad(v) + transpose(grad(v)))"))
    yield "anelastic", system(tsg.anelastic_problem())
    yield "solar_named", tsg.solar_named_system("product", 1e-3, 1e-6)
    yield "solar_named_grouped", tsg.solar_named_system("product", 1e-3, 1e-6,
                                                        pressure="grouped")
    yield "solar_named_true", tsg.solar_named_system("true", 2e-4, 1e-6, pressure="grouped")
    for visc in ("div(T)", "div(grad(v) + trans(grad(v)))", "lap(v) + grad(div(v))"):
        yield f"viscous_{visc}", system(tsg.viscous_problem(visc))
    yield "viscous_div(T)_dtT", system(tsg.viscous_problem("div(T)", dt_T=True))
    for bc in ("conducting", "pinned"):
        for pressure in ("grouped", "product"):
            yield f"ct_{bc}_{pressure}", tsg.ct_system(bc=bc, pressure=pressure)
    yield "ct_A", tsg.ct_system(formulation="A")
    poly = PolytropeBackground(r_inner=0.71, r_outer=0.985, delta=1e-6)
    f = (0.985 / r) ** 3
    B0 = Background.Vector("B0", v_r=2 * sp.cos(theta) * f, v_theta=sp.sin(theta) * f,
                           v_phi=0)
    for strain in (False, True):
        for Ek in (0.0, 7e-6):
            h = make_polytrope_system(poly, m_val=8, Ek=Ek, with_grad_lnrho_strain=strain)
            yield f"fig2_Ek{Ek}_strain{strain}", h
            if Ek:
                for form in ("b", "A"):
                    for bc in ("conducting", "pinned"):
                        yield (f"fig2_mhd_{form}_{bc}_strain{strain}",
                               add_magnetic_field(h, B0, Le=1e-3, eta=Ek, bc=bc,
                                                  formulation=form))
                yield (f"fig2_mhd_b_sol_strain{strain}",
                       add_magnetic_field(h, B0, Le=1e-3, eta=Ek, solenoidal=True))
    yield "fig2_ball", make_polytrope_system(poly, m_val=8, Ek=7e-6, geometry="ball")
    yield "fig2_Le", make_polytrope_system(poly, m_val=8, Ek=7e-6, Le=1e-3)
    for case in mf.CASES:
        for geometry in ("shell", "ball"):
            for form in ("source", "evolution"):
                pb, _, _ = mf.build_problem(case, geometry, 1, form)
                yield f"mf_{case.name}_{geometry}_{form}", system(pb)


def _classify(system, m=1, mimetic=True, geometry="shell", **kw):
    return classify(system, walk_equation_system(system, m), geometry=geometry,
                    mimetic=mimetic, **kw)


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mimetic", [True, False])
@pytest.mark.parametrize("m", [0, 1])
def test_boussinesq_is_a_saddle_point(mimetic, m):
    rec = _classify(boussinesq(), m, mimetic)
    assert isinstance(rec, ClassificationRecord)
    assert (rec.geometry, rec.mimetic, rec.m, rec.dt_less) == ("shell", mimetic, m, False)
    assert rec.rank == {"v": 1, "T": 0, "p": 0}
    assert rec.placement == {"v": "face", "T": "center", "p": "center"}
    assert {e: i["class"] for e, i in rec.equations.items()} == {
        "momentum": "evolution", "energy": "evolution", "continuity": "saddle_point"}
    assert rec.principal == {"momentum": "v", "energy": "T", "continuity": "p"}
    assert rec.multipliers == {"p": "continuity"}
    assert rec.constraints == {"v": "saddle_point"}
    assert rec.involutions == {} and rec.definitions == {}
    cont = rec.equations["continuity"]
    assert (cont["vector"], cont["multiplier"], cont["weight"]) == ("v", "p", None)
    assert cont["div_sign"] == 1 and cont["div_outer"] is None
    assert all(t.kind == TASK_GRAD_SCALAR and t.equation_name == "momentum"
               for t in cont["grad_tasks"])
    assert cont["gauge"] == ("point" if m == 0 else None)
    assert rec.gauges == {"p": cont["gauge"]}
    assert rec.involution_treatment == (
        INVOLUTION_DROPPED if mimetic else INVOLUTION_SADDLE_POINT)
    assert rec.divergence_cleaning == {}


def test_declared_gauge_unknown_takes_the_integral_gauge():
    assert _classify(boussinesq(gauge=True), 0).gauges == {"p": "integral"}
    assert _classify(boussinesq(gauge=True), 2).gauges == {"p": None}


@pytest.mark.parametrize("mimetic", [True, False])
def test_anelastic_weighted_divergence(mimetic):
    rec = _classify(anelastic(), 1, mimetic)
    cont = rec.equations["continuity"]
    assert cont["class"] == "saddle_point" and cont["multiplier"] == "p"
    assert isinstance(cont["weight"], SymPyComponent)
    assert sp.simplify(cont["weight"].expr - (1 - r ** 2 / 2) ** 2) == 0
    assert rec.constraints == {"v": "saddle_point"}


@pytest.mark.parametrize("geometry", ["shell", "ball"])
@pytest.mark.parametrize("m", [0, 1])
def test_written_involution(geometry, m):
    system = induction(ball=geometry == "ball")
    mimetic = _classify(system, m, True, geometry)
    assert mimetic.equations["solenoidal"] == {
        "class": "involution", "vector": "b", "evolution": "induction", "dropped": True}
    assert "solenoidal" not in mimetic.principal
    assert mimetic.constraints == {"v": "saddle_point", "b": "involution"}
    assert mimetic.involutions == {"b": "induction"}
    assert mimetic.involution_treatment == INVOLUTION_DROPPED
    assert mimetic.divergence_cleaning == {}
    cleaned = _classify(system, m, False, geometry)
    assert cleaned.equations["solenoidal"] == {
        "class": "involution", "vector": "b", "evolution": "induction", "dropped": False}
    for key in ("placement", "rank", "principal", "multipliers", "involutions",
                "constraints", "definitions"):
        assert getattr(cleaned, key) == getattr(mimetic, key)
    assert cleaned.involution_treatment == INVOLUTION_SADDLE_POINT
    walls = ("r=1",) if geometry == "ball" else ("r=0", "r=1")
    assert cleaned.divergence_cleaning == {"b": DivergenceCleaning(
        vector="b", multiplier="__div_b", evolution="induction", constraint="solenoidal",
        gauge="integral" if m == 0 else None, walls=walls)}
    assert cleaned.divergence_cleaning["b"].placement == "center"
    assert cleaned.divergence_cleaning["b"].rank == 0
    assert cleaned.gauges == {"p": "point" if m == 0 else None,
                              "__div_b": "integral" if m == 0 else None}


@pytest.mark.parametrize("diffusion", ["lap", "curlcurl", "none"])
def test_implied_involution(diffusion):
    system = induction(written=False, diffusion=diffusion)
    rec = _classify(system, 1, True)
    assert rec.involutions == {"b": "induction"}
    assert rec.constraints == {"v": "saddle_point", "b": "involution"}
    assert set(rec.equations) == {"momentum", "continuity", "induction"}
    cleaned = _classify(system, 0, False)
    assert cleaned.divergence_cleaning["b"].constraint is None
    assert cleaned.divergence_cleaning["b"].gauge == "integral"


def test_vector_potential_is_a_saddle_point():
    rec = _classify(vector_potential(), 0)
    assert rec.constraints == {"v": "saddle_point", "A": "saddle_point"}
    assert rec.multipliers == {"p": "continuity", "psi": "coulomb_gauge"}
    assert rec.involutions == {}
    assert rec.gauges == {"p": "integral", "psi": "integral"}
    assert _classify(vector_potential(), 0, False).divergence_cleaning == {}


@pytest.mark.parametrize("mimetic", [True, False])
def test_tensor_definition(mimetic):
    rec = _classify(tensor_definition(), 1, mimetic)
    d = rec.equations["def"]
    assert d["class"] == "definition" and d["tensor"] == "T" and d["vector"] == "v"
    assert d["identity"].unknown_name == "T"
    assert [ab for _t, ab in d["terms"]] == [(-1, -1)]
    assert rec.definitions == {"T": "def"}
    assert rec.principal["def"] == "T"
    assert rec.placement["T"] == "tensor" and rec.rank["T"] == 2
    assert _classify(tensor_definition("grad(v)"), 1, mimetic).placement["T"] == "tensor9"


def test_tensor_evolution_keeps_the_tensor_placement():
    rec = _classify(tensor_definition(dt_T=True), 1)
    assert rec.equations["def"] == {"class": "evolution", "unknown": "T"}
    assert rec.definitions == {} and rec.placement["T"] == "tensor"


@pytest.mark.parametrize("mimetic", [True, False])
def test_dt_less_rule(mimetic):
    rec = _classify(zhang(with_dt=False), 1, mimetic)
    assert rec.dt_less is True
    assert rec.equations["momentum"] == {"class": "evolution", "unknown": "v",
                                         "implicit_mass": True}
    assert rec.equations["continuity"]["class"] == "saddle_point"
    with_dt = _classify(zhang(), 1, mimetic)
    assert with_dt.dt_less is False
    assert with_dt.equations["momentum"] == {"class": "evolution", "unknown": "v"}


@pytest.mark.parametrize("mimetic", [True, False])
@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_refusals(name, mimetic):
    build, *fragments = REFUSALS[name]
    with pytest.raises(NotImplementedError) as info:
        _classify(build(), 1, mimetic)
    msg = str(info.value)
    assert msg.startswith("constraints:")
    for fragment in fragments:
        assert fragment in msg, (fragment, msg)


def test_refusal_message_prefix_follows_the_core():
    with pytest.raises(NotImplementedError, match="^staggered: equation 'relation'"):
        _classify(refusal_relation(), 1, core="staggered")


def test_auto_multiplier_name_must_be_free():
    system = refusal_multiplier_name_taken()
    assert _classify(system, 1, True).involutions == {"b": "induction"}
    with pytest.raises(NotImplementedError, match="already has an unknown named '__div_b'"):
        _classify(system, 1, False)


def test_geometry_is_checked():
    with pytest.raises(ValueError, match="geometry"):
        _classify(zhang(), 1, geometry="disk")


# ---------------------------------------------------------------------------
# parity with the C-grid classifier
# ---------------------------------------------------------------------------

def _walks(system):
    for m in (0, 1, 2):
        for aux in (True, False):
            try:
                aug = augment_equation_system(system, prefer_aux_free=aux)
                yield aug, walk_equation_system(aug, m)
            except (NotImplementedError, ValueError, TypeError):
                continue


def _outcome(fn, *args, **kw):
    try:
        return fn(*args, **kw), None
    except NotImplementedError as exc:
        return None, str(exc)


def test_agrees_with_the_staggered_classifier():
    """On every reference and phase-one system, at m = 0, 1, 2 and both
    augmentation modes, the two classifiers refuse the same systems with
    the same message and record the same placement, equations,
    principal, multipliers, involutions, constraints and definitions;
    the non-mimetic record differs only in the 'dropped' flag of a
    written involution."""
    from spdee.backends.fd2.staggered_general import classify_constraints
    seen = set()
    for name, system in list(reference_systems()) + list(dump_systems()):
        for aug, tasks in _walks(system):
            ref, ref_error = _outcome(classify_constraints, aug, tasks)
            rec, error = _outcome(classify, aug, tasks, geometry="shell", mimetic=True,
                                  core="staggered")
            assert error == ref_error, name
            cleaned, cleaned_error = _outcome(classify, aug, tasks, geometry="shell",
                                              mimetic=False, core="staggered")
            if ref is None:
                assert cleaned_error == ref_error, name
                seen.add("refused")
                continue
            if cleaned is None:
                assert "already has an unknown named" in cleaned_error, name
                seen.add("name_taken")
                continue
            assert rec.staggered_view() == ref, name
            for key, value in cleaned.staggered_view().items():
                if key != "equations":
                    assert value == ref[key], (name, key)
            for eq, info in cleaned.equations.items():
                expected = dict(ref["equations"][eq])
                if expected["class"] == "involution":
                    expected["dropped"] = False
                assert info == expected, (name, eq)
                seen.add(info["class"])
            if rec.dt_less:
                seen.add("dt_less")
    assert seen == {"refused", "evolution", "saddle_point", "involution", "definition",
                    "dt_less", "name_taken"}
