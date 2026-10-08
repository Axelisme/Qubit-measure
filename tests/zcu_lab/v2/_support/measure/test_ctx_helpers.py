"""Public fixed-parameter frequency/flux range policies."""

from __future__ import annotations

from collections.abc import Callable, Generator
from types import SimpleNamespace

import pytest
from simpleeval import simple_eval
from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext, RoleBinding
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import (
    ComponentSchema,
    RoleResolutionError,
    RoleSpec,
    component_registry,
)
from zcu_tools.resources.entry.schema import PointDocument

from zcu_lab.v2._support.measure.ctx_helpers import (
    proper_flux_range,
    proper_qub_freq_range,
    proper_res_freq_range,
)
from zcu_lab.v2._support.measure.seeds import res_freq_range


class RangeSensor(ComponentSchema):
    freq: YamlValue = None
    kappa: YamlValue = None


class RangeDriver(RangeSensor):
    flux_half: YamlValue = None
    flux_int: YamlValue = None


RangeFactory = Callable[[dict[str, YamlMap]], ParameterContext]


@pytest.fixture
def range_factory() -> Generator[RangeFactory]:
    before = component_registry.kinds()
    component_registry.register("range/sensor", RangeSensor)
    component_registry.register("range/drive", RangeDriver)

    def make(overrides: dict[str, YamlMap]) -> ParameterContext:
        components: YamlMap = {
            "R1": {"kind": "range/sensor", **overrides.get("R1", {})},
            "Q1": {"kind": "range/drive", **overrides.get("Q1", {})},
        }
        point = PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-04T00:00:00Z"},
                "components": components,
            }
        )
        return ParameterContext(
            point=point,
            library=LibrarySnapshot(
                ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
            ),
            target=EntryDraft(entry_id="00000000-0000-0000-0000-000000000001"),
            roles=RoleBinding(
                "Q1",
                {
                    "qubit": RoleSpec("range/drive"),
                    "resonator": RoleSpec("range/sensor"),
                },
                {"qubit": "Q1", "resonator": "R1"},
            ),
        )

    try:
        yield make
    finally:
        component_registry.unregister("range/drive")
        component_registry.unregister("range/sensor")
        assert component_registry.kinds() == before


def test_res_freq_range_uses_eval_value_when_parameters_present(
    range_factory: RangeFactory,
) -> None:
    sweep = proper_res_freq_range(
        range_factory({"R1": {"freq": 5500.0, "kappa": 10.0}}), 301
    )
    assert isinstance(sweep, SweepValue)
    assert isinstance(sweep.start, EvalValue)
    assert isinstance(sweep.stop, EvalValue)
    assert sweep.start.expr == "resonator.freq - 1.5 * resonator.kappa"
    assert sweep.start.resolved is None
    assert sweep.stop.expr == "resonator.freq + 1.5 * resonator.kappa"
    assert sweep.stop.resolved is None
    assert sweep.expts == 301


def test_res_freq_range_falls_back_to_scalar_without_calibration(
    range_factory: RangeFactory,
) -> None:
    sweep = proper_res_freq_range(range_factory({}), 101)
    assert sweep.start == 5750.0
    assert sweep.stop == 7250.0


def test_freq_range_span_factor_one_omits_coefficient(
    range_factory: RangeFactory,
) -> None:
    sweep = proper_res_freq_range(
        range_factory({"R1": {"freq": 6000.0, "kappa": 20.0}}), 101, span_factor=1.0
    )
    assert isinstance(sweep.start, EvalValue)
    assert sweep.start.expr == "resonator.freq - resonator.kappa"
    assert sweep.start.resolved is None


def test_qub_freq_range_uses_qubit_parameter_paths(range_factory: RangeFactory) -> None:
    sweep = proper_qub_freq_range(
        range_factory({"Q1": {"freq": 4200.0, "kappa": 5.0}}), 201, span_factor=2.0
    )
    assert isinstance(sweep.start, EvalValue)
    assert sweep.start.expr == "qubit.freq - 2.0 * qubit.kappa"
    assert sweep.start.resolved is None
    assert sweep.expts == 201


@pytest.mark.parametrize("calibration", [{"freq": 5500.0}, {"kappa": 10.0}])
def test_resonator_partial_calibration_keeps_the_existing_source_live(
    range_factory: RangeFactory, calibration: YamlMap
) -> None:
    sweep = proper_res_freq_range(range_factory({"R1": calibration}), 21)
    assert isinstance(sweep.start, EvalValue)
    assert isinstance(sweep.stop, EvalValue)
    before: list[float] = []
    after: list[float] = []
    original = SimpleNamespace(**calibration)
    doubled = SimpleNamespace(
        **{
            key: value * 2
            for key, value in calibration.items()
            if isinstance(value, float)
        }
    )
    for edge in (sweep.start, sweep.stop):
        first = simple_eval(edge.expr, names={"resonator": original})
        second = simple_eval(edge.expr, names={"resonator": doubled})
        assert isinstance(first, (int, float))
        assert isinstance(second, (int, float))
        before.append(float(first))
        after.append(float(second))
    if "freq" in calibration:
        assert after[0] - before[0] == pytest.approx(5500.0)
        assert after[1] - before[1] == pytest.approx(5500.0)
    else:
        assert after[1] - after[0] == pytest.approx(2 * (before[1] - before[0]))


def test_flux_range_extrapolates_past_calibrated_points(
    range_factory: RangeFactory,
) -> None:
    sweep = proper_flux_range(
        range_factory({"Q1": {"flux_half": 1e-3, "flux_int": 3e-3}}), 101
    )
    assert isinstance(sweep.start, EvalValue)
    assert isinstance(sweep.stop, EvalValue)
    assert sweep.start.expr == "1.1 * qubit.flux_int - 0.1 * qubit.flux_half"
    assert sweep.start.resolved is None
    assert sweep.stop.expr == "1.1 * qubit.flux_half - 0.1 * qubit.flux_int"
    assert sweep.expts == 101


def test_flux_range_falls_back_when_calibration_absent(
    range_factory: RangeFactory,
) -> None:
    sweep = proper_flux_range(range_factory({}), 101)
    assert sweep.start == -4e-3
    assert sweep.stop == 4e-3


@pytest.mark.parametrize(
    "calibration", [{"flux_half": 1e-3}, {"flux_int": None, "flux_half": 1e-3}]
)
def test_partial_or_null_flux_calibration_uses_original_fallback(
    range_factory: RangeFactory, calibration: YamlMap
) -> None:
    sweep = proper_flux_range(range_factory({"Q1": calibration}), 11)
    assert sweep.start == -4e-3
    assert sweep.stop == 4e-3


@pytest.mark.parametrize("value", [True, "5500", [], {}, float("inf"), float("nan")])
@pytest.mark.parametrize("field", ["freq", "kappa"])
def test_present_invalid_resonator_scalar_does_not_use_fallback(
    range_factory: RangeFactory, field: str, value: YamlValue
) -> None:
    with pytest.raises((TypeError, ValueError)):
        proper_res_freq_range(range_factory({"R1": {field: value}}), 11)


@pytest.mark.parametrize("field", ["flux_half", "flux_int"])
def test_present_invalid_flux_scalar_does_not_use_fallback(
    range_factory: RangeFactory, field: str
) -> None:
    with pytest.raises((TypeError, ValueError)):
        proper_flux_range(range_factory({"Q1": {field: "invalid"}}), 11)


def test_range_seed_uses_the_fixed_parameter_capture(
    range_factory: RangeFactory,
) -> None:
    context = range_factory({"R1": {"freq": 5500.0, "kappa": 10.0}})
    environment = SessionEnv(soc=None, soccfg=None, parameters=context)
    sweep = res_freq_range(expts=21).resolve(environment)
    assert isinstance(sweep.start, EvalValue)
    assert sweep.start.expr == "resonator.freq - 1.5 * resonator.kappa"
    assert sweep.start.resolved is None
    assert sweep.expts == 21


def test_unknown_role_does_not_become_uncalibrated_fallback(
    range_factory: RangeFactory,
) -> None:
    context = range_factory({})
    direct = ParameterContext(
        point=context.document(),
        library=context.library,
        target=context.target,
        roles=None,
    )
    with pytest.raises(RoleResolutionError):
        proper_res_freq_range(direct, 11)
