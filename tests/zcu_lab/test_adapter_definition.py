"""Registry-wide contract tests for context-free measure cfg definitions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import NoneResetCfg, PulseReadoutSpec, PulseSpec
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalyzeRequest,
    ExpAdapterProtocol,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.cfg_binding import (
    MeasureCfgBindingHost,
    MeasureCfgBindings,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.cfg import CfgSectionSpec, EvalValue, LiteralSpec, SweepValue
from zcu_tools.gui.cfg.binding.observation import CfgNodeObservation
from zcu_tools.gui.cfg.binding.ports import ResolvedReference
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgResolution,
    CfgResource,
    CfgStatus,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from zcu_lab.components import JpaSchema, QubitSchema, ResonatorSchema
from zcu_lab.definitions import register_all
from zcu_lab.v2.singleshot.mist.power.core import PowerExp


@pytest.mark.parametrize("log_scale", [False, True])
def test_registry_analysis_input_controls_mist_power_scale(
    monkeypatch, log_scale: bool
) -> None:
    adapter = _registry().create("singleshot/mist/power")
    environment = _ctx(rich=True)
    params = adapter.get_analyze_params(object(), environment)
    assert getattr(params, "log_scale", None) is False
    selected = replace(params, log_scale=log_scale)
    analyze = MagicMock()
    monkeypatch.setattr(PowerExp, "analyze", analyze)
    adapter.analyze(
        AnalyzeRequest(
            run_result=object(),
            analyze_params=selected,
            parameters=environment.parameters,
        ),
        plots=Plots(NonPresentingHost()),
    )
    options = analyze.call_args.args[1]
    assert options.log_scale is log_scale


def _registry() -> Registry:
    registry = Registry()
    register_all(registry)
    return registry


def _flux_adapter_names() -> list[str]:
    registry = _registry()
    names = []
    for name in registry.list_names():
        adapter = registry.create(name)
        assert isinstance(adapter, BaseAdapter)
        dev = type(adapter).cfg_definition().spec.fields.get("dev")
        if isinstance(dev, CfgSectionSpec) and {
            "flux_dev",
            "jpa_flux_dev",
        }.intersection(dev.fields):
            names.append(name)
    return names


@pytest.mark.parametrize("name", _flux_adapter_names())
def test_registered_flux_cfg_requires_explicit_unit(name: str) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)
    cfg_model = adapter.exp_cls().persistence_spec().cfg_type
    with pytest.raises(ValueError, match="flux_unit"):
        cfg_model.model_validate({})


@pytest.mark.parametrize("name", _flux_adapter_names())
def test_registered_flux_form_edits_unit_without_device_observations(name: str) -> None:
    adapter = _registry().create(name)
    environment = _ctx(rich=True)
    parameters = environment.parameters
    assert parameters is not None
    point = parameters.document()
    point.components["R1"] = point.components["R1"].model_copy(
        update={"ext": {"time_of_fly": 0.45, "default_probe_length": 1.0}}
    )
    environment = replace(
        environment,
        parameters=ParameterContext(
            point=point,
            library=parameters.library,
            target=parameters.target,
            roles=None,
        ).bind_aliases(
            (
                AliasRow("qubit", "Q1"),
                AliasRow("resonator", "$qubit.resonator"),
                AliasRow("jpa", "$resonator.amplifier"),
                AliasRow("flux_dev", "flux_yoko"),
            ),
            {},
        ),
    )
    host = MagicMock(spec=MeasureCfgBindingHost)
    host.list_device_names.return_value = ["flux_yoko"]
    host.arb_waveforms.list_data_keys.return_value = []
    bindings = MeasureCfgBindings(host)
    resource = CfgResource(
        defaults=lambda: adapter.make_default_cfg(environment),
        resolution=lambda: bindings.snapshot(
            (),
            captured_values={},
            parameters=environment.parameters,
        ),
    )
    try:
        observed = resource.observe()
        accepted = resource.accept(observed.ref.revision)
        assert accepted.values.get("flux_unit") == "A"
        dev = accepted.values.get("dev")
        for unit in ("V", "native"):
            observed = resource.edit(
                observed.ref.revision, (CfgEdit(("flux_unit",), unit),)
            )
            accepted = resource.accept(observed.ref.revision)
            assert accepted.values["flux_unit"] == unit
            assert accepted.values.get("dev") == dev
        invalid = resource.edit(observed.ref.revision, (CfgEdit(("flux_unit",), "mA"),))
        assert invalid.status is CfgStatus.INVALID
    finally:
        resource.revoke()


class _EmptyCatalog:
    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return ()

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return None


def _empty_resolution() -> CfgResolution:
    return CfgResolution(
        (),
        lambda expression: 0.0,
        lambda source_id: (),
        _EmptyCatalog(),
        lambda name: 0.0,
        lambda expression: None,
    )


def _form_rows(
    node: CfgNodeObservation, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    """Yield editable node paths in the depth-first order the cfg form renders."""
    for key, child in node.children.items():
        if isinstance(child.spec, LiteralSpec):
            continue
        child_path = (*path, key)
        yield child_path
        yield from _form_rows(child, child_path)


@pytest.mark.parametrize("name", _flux_adapter_names())
def test_registered_flux_form_lists_unit_right_after_flux_device(name: str) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)
    resource = CfgResource(
        defaults=lambda: adapter.make_default_cfg(_ctx(rich=False)),
        resolution=_empty_resolution,
    )
    try:
        rows = list(_form_rows(resource.observe().tree))
        flux_device = next(
            row for row in rows if row in {("dev", "flux_dev"), ("dev", "jpa_flux_dev")}
        )
        assert rows[rows.index(flux_device) + 1] == ("flux_unit",)
    finally:
        resource.revoke()


def _pulse_raw(*, freq: float, gain: float) -> dict[str, object]:
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": 0.05},
        "ch": 3,
        "nqz": 2,
        "freq": freq,
        "mixer_freq": 1000.0,
        "gain": gain,
    }


def _readout_raw() -> dict[str, object]:
    return {
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 1.0},
            "ch": 1,
            "nqz": 2,
            "freq": 6100.0,
            "gain": 0.2,
        },
        "ro_cfg": {
            "ro_ch": 2,
            "ro_freq": 6100.0,
            "ro_length": 1.0,
            "trig_offset": 0.5,
        },
    }


def _ctx(*, rich: bool) -> SessionEnv:
    if not rich:
        return SessionEnv(parameters=None, soc=None, soccfg=None)
    point = PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="1.0",
        general=PointGeneral(created_at="2026-10-09T00:00:00Z"),
        components={
            "Q1": QubitSchema.model_validate(
                {
                    "kind": "qubit/fluxonium",
                    "freq": 4200.0,
                    "kappa": 10.0,
                    "t1": 12.0,
                    "resonator": "R1",
                    "flux_half": 0.001,
                    "flux_int": 0.003,
                    "wiring": {"L01": {"ch": 3}},
                    "module": {
                        "x180": "Q1.pulses.cal",
                        "x90": "Q1.pulses.half",
                        "readout": "Q1.readout.cal",
                        "reset": "Q1.reset.cal",
                    },
                    "ext": {
                        "best_ro_freq": 6100.0,
                        "best_ro_gain": 0.2,
                        "best_ro_length": 1.0,
                        "readout_cal": {
                            "g_center": 0.0,
                            "e_center": 1.0,
                            "radius": 0.5,
                        },
                    },
                }
            ),
            "R1": ResonatorSchema.model_validate(
                {
                    "kind": "resonator",
                    "freq": 6000.0,
                    "kappa": 20.0,
                    "amplifier": "J1",
                    "wiring": {"ch": 1, "ro_ch": 2},
                    "ext": {"time_of_fly": 0.45},
                }
            ),
            "J1": JpaSchema(
                kind="amplifier/jpa", pump_freq=12000.0, pump_power=-10.0, flux=0.002
            ),
        },
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="1.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={
                        "cal": PulseSpec.model_validate(
                            _pulse_raw(freq=4200.0, gain=0.4)
                        ),
                        "half": PulseSpec.model_validate(
                            _pulse_raw(freq=4200.0, gain=0.2)
                        ),
                    },
                    readout={"cal": PulseReadoutSpec.model_validate(_readout_raw())},
                    reset={"cal": NoneResetCfg()},
                )
            },
        )
    )
    parameters = ParameterContext(
        point=point, library=library, target=EntryDraft(entry_id="entry"), roles=None
    ).bind_aliases(
        (
            AliasRow("qubit", "Q1"),
            AliasRow("resonator", "$qubit.resonator"),
            AliasRow("jpa", "$resonator.amplifier"),
        ),
        {},
    )
    return SessionEnv(parameters=parameters, soc=None, soccfg=None)


def _specified_defaults() -> list[str]:
    registry = _registry()
    return [
        name
        for name in registry.list_names()
        if name.startswith("jpa/") or name == "singleshot/ac_stark"
    ]


def _sweep_centers(tree: CfgNodeObservation) -> Iterator[tuple[str, float]]:
    value = tree.value
    if (
        isinstance(value, SweepValue)
        and isinstance(value.start, EvalValue)
        and isinstance(value.stop, EvalValue)
    ):
        start, stop = value.start.resolved, value.stop.resolved
        if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
            raise AssertionError("Calibrated sweep must resolve real endpoints")
        yield value.start.expr, (start + stop) / 2
    for child in tree.children.values():
        yield from _sweep_centers(child)


def _assert_calibration_moves_sweep_centers(
    before: CfgNodeObservation, after: CfgNodeObservation
) -> None:
    # These are fixture calibration changes, not a copy of adapter sweep spans.
    shifts = {
        "$qubit.freq": 200.0,
        "$resonator.freq": 200.0,
        "$jpa.pump_freq": 1000.0,
        "$jpa.pump_power": 2.0,
        "$jpa.flux": 0.002,
    }
    centers = dict(_sweep_centers(after))
    checked = 0
    for expression, center in _sweep_centers(before):
        for path, shift in shifts.items():
            if path in expression:
                assert centers[expression] == pytest.approx(center + shift)
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("name", _specified_defaults())
def test_canonical_defaults_resolve_without_md_and_refresh_fixed_alias_sources(
    name: str,
) -> None:
    adapter = _registry().create(name)
    environment = _ctx(rich=False)
    host = MagicMock(spec=MeasureCfgBindingHost)
    host.list_device_names.return_value = ["flux_yoko"]
    host.arb_waveforms.list_data_keys.return_value = []
    bindings = MeasureCfgBindings(host)
    resource = CfgResource(
        defaults=lambda: adapter.make_default_cfg(environment),
        resolution=lambda: bindings.snapshot(
            (), captured_values={}, parameters=environment.parameters
        ),
    )
    try:
        missing = resource.observe()
        assert missing.status is CfgStatus.INVALID
        assert any(
            "$qubit" in item.message
            or "$resonator" in item.message
            or "$jpa" in item.message
            for item in missing.diagnostics
        )
        inputs = resource.snapshot_inputs()
        environment = _ctx(rich=True)
        parameters = environment.parameters
        assert parameters is not None
        rows = (
            AliasRow("qubit", "Q1"),
            AliasRow("resonator", "$qubit.resonator"),
            AliasRow("jpa", "$resonator.amplifier"),
            AliasRow("flux_dev", "flux_yoko"),
        )
        environment = replace(
            environment,
            parameters=parameters.bind_aliases(rows, {}),
        )
        calibrated = resource.refresh(missing.ref.revision)
        assert calibrated.status is CfgStatus.VALID, calibrated.diagnostics
        assert resource.snapshot_inputs() == inputs
        accepted = resource.accept(calibrated.ref.revision)
        point = parameters.document()
        point.components["Q1"] = point.components["Q1"].model_copy(
            update={"freq": 4400.0}
        )
        point.components["J1"] = point.components["J1"].model_copy(
            update={"pump_freq": 13000.0, "pump_power": -8.0, "flux": 0.004}
        )
        point.components["R1"] = point.components["R1"].model_copy(
            update={"freq": 6200.0}
        )
        updated_parameters = ParameterContext(
            point=point,
            library=parameters.library,
            target=parameters.target,
            roles=None,
        ).bind_aliases(rows, {})
        environment = replace(environment, parameters=updated_parameters)
        updated = resource.refresh(calibrated.ref.revision)
        assert updated.status is CfgStatus.VALID, updated.diagnostics
        updated_values = resource.accept(updated.ref.revision).values
        assert updated_values != accepted.values
        _assert_calibration_moves_sweep_centers(calibrated.tree, updated.tree)
        environment = replace(
            environment,
            parameters=parameters.bind_aliases(
                (
                    AliasRow("qubit", ""),
                    AliasRow("resonator", "$qubit.resonator"),
                    AliasRow("jpa", "$resonator.amplifier"),
                    AliasRow("flux_dev", "flux_yoko"),
                ),
                {},
            ),
        )
        unbound = resource.refresh(updated.ref.revision)
        assert unbound.status is CfgStatus.INVALID
        assert any("qubit" in item.message for item in unbound.diagnostics)
        environment = replace(environment, parameters=updated_parameters)
        rebound = resource.refresh(unbound.ref.revision)
        assert rebound.status is CfgStatus.VALID
        assert resource.accept(rebound.ref.revision).values == updated_values
        for component, fields in {
            "Q1": {"freq": None},
            "R1": {"freq": None},
            "J1": {"pump_freq": None, "pump_power": None, "flux": None},
        }.items():
            point.components[component] = point.components[component].model_copy(
                update=fields
            )
        environment = replace(
            environment,
            parameters=ParameterContext(
                point=point,
                library=parameters.library,
                target=parameters.target,
                roles=None,
            ).bind_aliases(rows, {}),
        )
        uncalibrated = resource.refresh(rebound.ref.revision)
        assert uncalibrated.status is CfgStatus.INVALID
        assert any(
            "$qubit" in item.message
            or "$resonator" in item.message
            or "$jpa" in item.message
            for item in uncalibrated.diagnostics
        )
    finally:
        resource.revoke()


@pytest.mark.parametrize("name", _registry().list_names())
def test_registered_adapter_conforms_to_framework(name: str) -> None:
    adapter = _registry().create(name)

    assert isinstance(adapter, ExpAdapterProtocol)
    assert isinstance(adapter.capabilities, AdapterCapabilities)


@pytest.mark.parametrize("name", _registry().list_names())
def test_registered_adapter_definition_has_context_free_shape(name: str) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)

    definition = type(adapter).cfg_definition()
    empty_cfg = adapter.make_default_cfg(_ctx(rich=False))
    rich_cfg = adapter.make_default_cfg(_ctx(rich=True))

    assert empty_cfg.spec == definition.spec
    assert rich_cfg.spec == definition.spec
    assert rich_cfg.spec == empty_cfg.spec


@pytest.mark.parametrize("name", _registry().list_names())
def test_registered_adapter_definition_is_reusable(name: str) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)

    definition = type(adapter).cfg_definition()
    ctx = _ctx(rich=True)

    first = definition.instantiate(ctx)
    second = definition.instantiate(ctx)
    expected = definition.instantiate(ctx)

    assert first == second == expected
    assert first.spec is not second.spec
    assert first.spec.fields is not second.spec.fields
    assert first.value is not second.value
    assert first.value.fields is not second.value.fields

    first.spec.fields.clear()
    first.value.fields.clear()
    assert second == expected
    assert definition.instantiate(ctx) == expected

    exposed_spec = definition.spec
    exposed_spec.fields.clear()
    assert definition.spec == expected.spec
