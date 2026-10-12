"""Registry contract for role-aware adapter defaults and captured lowering."""

import re
from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.cfg_model import (
    BathResetSpec,
    PulseResetSpec,
    PulseSpec,
    TwoPulseResetSpec,
)
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    is_custom_reference_key,
    read_value_path,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleResolutionError,
    RoleSpec,
    component_registry,
)
from zcu_tools.resources.entry.schema import PointDocument
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from zcu_lab.components import QubitSchema, ResonatorSchema
from zcu_lab.definitions import register_all


def _registry() -> Registry:
    registry = Registry()
    register_all(registry)
    return registry


def _role_adapters() -> list[str]:
    registry = _registry()
    return [
        name
        for name in registry.list_names()
        if name.startswith(("onetone/", "twotone/rabi/"))
        or name
        in {
            "lookback",
            "singleshot/ge",
            "singleshot/ac_stark",
            "twotone/freq",
            "twotone/ro_optimize/freq_gain",
            "twotone/t1",
            "twotone/t2ramsey",
            "twotone/t2echo",
        }
    ]


@pytest.fixture(autouse=True)
def component_kinds() -> Generator[None]:
    before = component_registry.kinds()
    added: list[str] = []
    for kind, model in [
        ("qubit/transmon", QubitSchema),
        ("resonator", ResonatorSchema),
    ]:
        if kind not in before:
            component_registry.register(kind, model)
            added.append(kind)
    existing_roles: dict[str, RoleSpec] = {}
    added_roles: list[str] = []
    for role, spec in [
        ("qubit", RoleSpec("qubit/*")),
        ("resonator", RoleSpec("resonator")),
    ]:
        try:
            existing_roles[role] = component_registry.roles.get(role)
        except ValueError:
            component_registry.roles.register(role, spec)
            added_roles.append(role)
    try:
        yield
    finally:
        for role in reversed(added_roles):
            component_registry.roles.unregister(role)
        for role, spec in existing_roles.items():
            assert component_registry.roles.get(role) == spec
        for kind in reversed(added):
            component_registry.unregister(kind)
        assert component_registry.kinds() == before


def _source(*, calibrated: bool, null_calibrations: bool = False) -> ParameterContext:
    components: dict[str, object] = {}
    libraries: dict[str, object] = {}
    for index in [1, 2]:
        qubit, resonator = f"Q{index}", f"R{index}"
        pulse = {
            "waveform": {"style": "const", "length": 0.05 * index},
            "ch": index,
            "nqz": 2,
            "freq": 4200.0 + 200 * index,
            "gain": 0.2 * index,
        }
        components[qubit] = {
            "kind": "qubit/transmon",
            "resonator": resonator,
            "freq": None if null_calibrations else pulse["freq"],
            "kappa": 10.0 * index,
            "t1": None if null_calibrations else 12.0 * index,
            "wiring": {"L01": {"ch": index}},
            "ext": {"best_ro_freq": 6000.0 + 100 * index},
            "module": (
                {
                    "x180": f"{qubit}.pulses.cal",
                    "x90": f"{qubit}.pulses.half",
                    "readout": f"{resonator}.readout.cal",
                }
                if calibrated
                else {}
            ),
        }
        components[resonator] = {
            "kind": "resonator",
            "freq": None if null_calibrations else 6000.0 + 100 * index,
            "kappa": 20.0 * index,
            "wiring": {"ch": index, "ro_ch": index},
            "ext": {
                "default_probe_length": None if null_calibrations else 2.0 * index,
                "time_of_fly": None if null_calibrations else 0.2 * index,
            },
        }
        libraries[qubit] = {
            "pulses": {"cal": pulse, "half": {**pulse, "gain": 0.1 * index}}
        }
        libraries[resonator] = {
            "readout": {
                "cal": {
                    "type": "readout/pulse",
                    "pulse_cfg": {**pulse, "freq": 6000.0 + 100 * index},
                    "ro_cfg": {
                        "ro_ch": index,
                        "ro_length": 1.0,
                        "trig_offset": 0.5,
                    },
                }
            }
        }
    return ParameterContext(
        point=PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-08T00:00:00Z"},
                "components": components,
            }
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument.model_validate(
                {
                    "format": "zcu.module-library",
                    "format_version": "1.0",
                    "components": libraries,
                }
            )
        ),
        target=SavedPoint(entry_id="contract-entry", point="working"),
        roles=None,
    )


@pytest.mark.parametrize("name", _role_adapters())
@pytest.mark.parametrize("calibrated", [False, True], ids=["inline", "library"])
def test_registered_role_adapter_lowers_one_template_on_fixed_captures(
    name: str, calibrated: bool
) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)
    source = _source(calibrated=calibrated)
    first = source.bind_aliases(
        (
            AliasRow("qubit", "Q1"),
            AliasRow("resonator", "$qubit.resonator"),
            AliasRow("flux_dev", "flux_yoko"),
        ),
        {},
    )
    second = source.bind_aliases(
        (
            AliasRow("qubit", "Q2"),
            AliasRow("resonator", "$qubit.resonator"),
            AliasRow("flux_dev", "flux_yoko"),
        ),
        {},
    )
    definition = type(adapter).cfg_definition()
    schema = definition.instantiate(SessionEnv(parameters=first, soc=None, soccfg=None))
    second_schema = definition.instantiate(
        SessionEnv(parameters=second, soc=None, soccfg=None)
    )
    assert definition.spec == schema.spec == second_schema.spec
    # SMART still supplies fresh optional/inline modules. Required scalar
    # calibrations and module leaves do not acquire numerical fallbacks.
    failures = {
        "onetone/flux_dep": (
            "sweep.flux.start",
            "Invalid parameter path '$qubit.flux_int': missing parameter",
        ),
        "twotone/t2ramsey": (
            "sweep.length.stop",
            "Invalid parameter path '$qubit.t2r': missing parameter",
        ),
        "twotone/t2echo": (
            "sweep.length.stop",
            "Invalid parameter path '$qubit.t2e': missing parameter",
        ),
        "singleshot/ac_stark": (
            (
                "modules.stark_pulse2.mixer_freq"
                if calibrated
                else "modules.stark_pulse1.waveform.length"
            ),
            (
                "Parameter path '$qubit.module.x180.mixer_freq' is not numeric"
                if calibrated
                else "Unset module slot: 'Q{index}.module.x180'"
            ),
        ),
    }
    if not calibrated:
        failures.update(
            {
                "twotone/rabi/amp_rabi": (
                    "modules.qub_pulse.waveform.length",
                    "Unset module slot: 'Q{index}.module.x180'",
                ),
                "twotone/rabi/len_rabi": (
                    "sweep.length.stop",
                    "Unset module slot: 'Q{index}.module.x180'",
                ),
            }
        )
    if name in failures:
        field, reason = failures[name]
        for index, cfg, capture in [(1, schema, first), (2, second_schema, second)]:
            with pytest.raises(
                RuntimeError, match=re.escape(f"Config field '{field}'")
            ) as error:
                schema_to_raw_dict(cfg, capture)
            assert str(error.value).endswith(reason.format(index=index))
        return
    raw_first = schema_to_raw_dict(schema, first)
    module_spec = definition.spec.fields["modules"]
    assert isinstance(module_spec, CfgSectionSpec)
    pulses: list[tuple[str, ReferenceSpec]] = []
    for key, spec in module_spec.fields.items():
        if isinstance(spec, ReferenceSpec):
            discriminator = spec.allowed[0].fields.get("type")
            if (
                isinstance(discriminator, LiteralSpec)
                and discriminator.value == "pulse"
            ):
                pulses.append((key, spec))
    for capture, cfg, raw in [
        (first, schema, raw_first),
        (second, second_schema, schema_to_raw_dict(second_schema, second)),
    ]:
        readout = read_value_path(cfg.value, "modules.readout")
        assert isinstance(readout, ReferenceValue)
        references = [readout]
        modules = raw["modules"]
        assert isinstance(modules, dict)
        for pulse_key, pulse_spec in pulses:
            pulse = read_value_path(cfg.value, f"modules.{pulse_key}")
            if pulse is None:
                assert pulse_spec.optional
            else:
                assert isinstance(pulse, ReferenceValue)
                references.append(pulse)
                lowered_pulse = modules[pulse_key]
                assert isinstance(lowered_pulse, dict)
                assert (
                    lowered_pulse["ch"] == capture.lookup("$qubit.wiring.L01.ch").value
                )
        for ref in references:
            if not is_custom_reference_key(ref.chosen_key):
                assert ref.chosen_key in {
                    "$qubit.module.x180",
                    "$qubit.module.x90",
                    "$qubit.module.readout",
                } and (
                    capture.library_key(ref.chosen_key)
                    == capture.lookup(ref.chosen_key).value
                )
        lowered_readout = modules["readout"]
        assert isinstance(lowered_readout, dict)
        readout_pulse, readout_ro = (
            lowered_readout["pulse_cfg"],
            lowered_readout["ro_cfg"],
        )
        assert isinstance(readout_pulse, dict) and isinstance(readout_ro, dict)
        assert readout_pulse["ch"] == capture.lookup("$resonator.wiring.ch").value
        assert readout_ro["ro_ch"] == capture.lookup("$resonator.wiring.ro_ch").value
    # Later target selection does not alter the original source or the template.
    source.bind_aliases(
        (
            AliasRow("qubit", "Q2"),
            AliasRow("resonator", "$qubit.resonator"),
            AliasRow("flux_dev", "flux_yoko"),
        ),
        {},
    )
    assert schema_to_raw_dict(schema, first) == raw_first
    assert (
        definition.instantiate(SessionEnv(parameters=first, soc=None, soccfg=None))
        == schema
    )


@pytest.mark.parametrize("name", _role_adapters())
@pytest.mark.parametrize("null_calibrations", [False, True])
def test_registered_readout_frequency_sweep_uses_captured_probe_timing(
    name: str, null_calibrations: bool
) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)
    modules_spec = type(adapter).cfg_definition().spec.fields["modules"]
    assert isinstance(modules_spec, CfgSectionSpec)
    for spec in modules_spec.fields.values():
        if isinstance(spec, ReferenceSpec):
            discriminator = spec.allowed[0].fields.get("type")
            if (
                isinstance(discriminator, LiteralSpec)
                and discriminator.value == "pulse"
            ):
                pytest.skip("adapter also calibrates a qubit-drive pulse")
    readout_spec = modules_spec.fields["readout"]
    assert isinstance(readout_spec, ReferenceSpec)
    pulse_spec = readout_spec.allowed[0].fields.get("pulse_cfg")
    if not isinstance(pulse_spec, CfgSectionSpec) or not isinstance(
        pulse_spec.fields["freq"], LiteralSpec
    ):
        pytest.skip("adapter does not own an inline readout-frequency sweep")
    capture = _source(
        calibrated=True, null_calibrations=null_calibrations
    ).bind_aliases(
        (
            AliasRow("qubit", "Q2"),
            AliasRow("resonator", "$qubit.resonator"),
            AliasRow("flux_dev", "flux_yoko"),
        ),
        {},
    )
    schema = adapter.make_default_cfg(
        SessionEnv(parameters=capture, soc=None, soccfg=None)
    )
    if null_calibrations:
        with pytest.raises(
            RuntimeError,
            match=re.escape("Config field 'modules.readout.ro_cfg.ro_length'"),
        ) as error:
            schema_to_raw_dict(schema, capture)
        assert str(error.value).endswith(
            "Parameter path '$resonator.ext.default_probe_length' is not numeric"
        )
        return
    if name == "onetone/flux_dep":
        with pytest.raises(
            RuntimeError, match=re.escape("Config field 'sweep.flux.start'")
        ) as error:
            schema_to_raw_dict(schema, capture)
        assert str(error.value).endswith(
            "Invalid parameter path '$qubit.flux_int': missing parameter"
        )
        return
    raw = schema_to_raw_dict(schema, capture)
    modules = raw["modules"]
    assert isinstance(modules, dict)
    readout = modules["readout"]
    assert isinstance(readout, dict)
    pulse, acquisition = readout["pulse_cfg"], readout["ro_cfg"]
    assert isinstance(pulse, dict) and isinstance(acquisition, dict)
    assert pulse["freq"] == acquisition["ro_freq"] == 0.0
    assert acquisition["ro_length"] == pytest.approx(3.9)
    assert acquisition["trig_offset"] == pytest.approx(0.45)


@pytest.mark.parametrize("name", _role_adapters())
def test_registered_role_adapter_rejects_missing_required_calibrations(
    name: str,
) -> None:
    adapter = _registry().create(name)
    capture = _source(calibrated=False, null_calibrations=True).bind_aliases(
        (AliasRow("qubit", "Q2"), AliasRow("resonator", "$qubit.resonator")), {}
    )
    schema = adapter.make_default_cfg(
        SessionEnv(parameters=capture, soc=None, soccfg=None)
    )
    # Keep the original null scalars and unset slots: lowering must identify
    # the first required calibration, not manufacture a runnable cfg.
    failures = {
        "lookback": ("modules.readout.pulse_cfg.freq", "$resonator.freq"),
        "onetone/freq": (
            "modules.readout.ro_cfg.ro_length",
            "$resonator.ext.default_probe_length",
        ),
        "onetone/power_dep": (
            "modules.readout.ro_cfg.ro_length",
            "$resonator.ext.default_probe_length",
        ),
        "onetone/flux_dep": (
            "modules.readout.ro_cfg.ro_length",
            "$resonator.ext.default_probe_length",
        ),
        "twotone/freq": ("modules.readout.pulse_cfg.freq", "$resonator.freq"),
        "twotone/rabi/amp_rabi": ("modules.qub_pulse.waveform.length", None),
        "twotone/rabi/len_rabi": ("modules.qub_pulse.freq", "$qubit.freq"),
        "twotone/ro_optimize/freq_gain": ("modules.qub_pulse.freq", "$qubit.freq"),
        "twotone/t1": ("modules.pi_pulse.freq", "$qubit.freq"),
        "twotone/t2ramsey": ("modules.pi2_pulse.freq", "$qubit.freq"),
        "twotone/t2echo": ("modules.pi_pulse.freq", "$qubit.freq"),
        "singleshot/ge": ("modules.probe_pulse.freq", "$qubit.freq"),
        "singleshot/ac_stark": ("modules.stark_pulse1.waveform.length", None),
    }
    field, path = failures[name]
    with pytest.raises(
        RuntimeError, match=re.escape(f"Config field '{field}'")
    ) as error:
        schema_to_raw_dict(schema, capture)
    expected = (
        f"Parameter path '{path}' is not numeric"
        if path is not None
        else "Unset module slot: 'Q2.module.x180'"
    )
    assert str(error.value).endswith(expected)


@pytest.mark.parametrize("name", _role_adapters())
def test_registered_adapter_does_not_hide_missing_aliases_or_source_faults(
    name: str,
) -> None:
    adapter = _registry().create(name)
    source = _source(calibrated=True)
    unbound = source.bind_aliases(
        (AliasRow("qubit", ""), AliasRow("resonator", "$qubit.resonator")), {}
    )
    for role in ("qubit", "resonator"):
        with pytest.raises(RoleResolutionError) as error:
            unbound.resolve_path(f"${role}.freq")
        assert error.value.role == role
    capture = source.bind_aliases(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator")), {}
    )
    unavailable = ParameterContext(
        point=capture.document(),
        library=capture.library,
        target=capture.target,
        roles=capture.roles,
        source_faults=(
            ContextSourceFault(
                document="point",
                path=Path("/contract/point.yaml"),
                error_type="ValueError",
                message="broken source",
            ),
        ),
    )
    if name in {"lookback", "onetone/freq", "onetone/power_dep"}:
        schema = adapter.make_default_cfg(
            SessionEnv(parameters=unavailable, soc=None, soccfg=None)
        )
        with pytest.raises(RuntimeError, match="point: broken source") as error:
            schema_to_raw_dict(schema, unavailable)
        cause = error.value.__cause__
        assert isinstance(cause, ContextSourceUnavailableError)
        assert unavailable.source_faults[0] in cause.faults
        return
    with pytest.raises(ContextSourceUnavailableError):
        adapter.make_default_cfg(
            SessionEnv(parameters=unavailable, soc=None, soccfg=None)
        )


_RESET_READERS = {
    "single-freq": "twotone/reset/single_tone/freq",
    "single-length": "twotone/reset/single_tone/length",
    "dual-freq": "twotone/reset/dual_tone/freq",
    "dual-length": "twotone/reset/dual_tone/length",
    "dual-power": "twotone/reset/dual_tone/power",
    "bath-length": "twotone/reset/bath/length",
    "bath-phase": "twotone/reset/bath/phase",
}
_RESET_REQUIRED = [
    ("single-freq", "reset_f"),
    ("single-freq", "reset_fwhm"),
    ("single-length", "reset_f"),
    ("dual-freq", "reset_f1"),
    ("dual-freq", "reset_f2"),
    ("dual-freq", "reset_gain1"),
    ("dual-freq", "reset_gain2"),
    ("dual-length", "reset_f1"),
    ("dual-length", "reset_f2"),
    ("dual-length", "reset_gain1"),
    ("dual-length", "reset_gain2"),
    ("dual-power", "reset_f1"),
    ("dual-power", "reset_f2"),
    ("bath-length", "reset_bath_freq"),
    ("bath-length", "reset_bath_gain"),
    ("bath-phase", "reset_bath_freq"),
    ("bath-phase", "reset_bath_gain"),
]


def _reset_seed_source() -> ParameterContext:
    """Two legal captures with different ext seeds and unrelated old slots."""
    source = _source(calibrated=True)
    point = source.document()
    library = source.library
    for index in (1, 2):
        qubit = f"Q{index}"
        component = QubitSchema.model_validate(point.components[qubit].model_dump())
        component.ext.update(
            {
                "reset_f": 3200.0 + 1000 * index,
                "reset_fwhm": 8.0,
                "reset_f1": 3300.0 + 1000 * index,
                "reset_f2": 3400.0 + 1000 * index,
                "reset_gain1": 0.21 if index == 1 else 0.41,
                "reset_gain2": 0.31 if index == 1 else 0.51,
                "reset_bath_freq": 6200.0 + 100 * index,
                "reset_bath_gain": 0.25 if index == 1 else 0.35,
            }
        )
        old = PulseSpec.model_validate(
            {
                "ch": 1,
                "nqz": 1,
                "freq": 4120.0,
                "gain": 0.7,
                "waveform": {"style": "const", "length": 0.9},
            }
        )
        single = PulseResetSpec(pulse_cfg=old)
        dual = TwoPulseResetSpec(
            pulse1_cfg=old,
            pulse2_cfg=old.model_copy(update={"freq": 4220.0, "gain": 0.8}),
        )
        bath = BathResetSpec(
            cavity_tone_cfg=old.model_copy(update={"freq": 6100.0}),
            qubit_tone_cfg=old,
            pi2_cfg=old.model_copy(update={"phase": 31.0}),
        )
        for slot, spec in (
            ("sideband", single),
            ("dual_tone", dual),
            ("bath_g", bath),
            ("bath_e", bath),
        ):
            root = f"{qubit}.reset.old_{slot}"
            component.module[slot] = root
            library = library.replaced_module(root, spec)
        point.components[qubit] = component
    return ParameterContext(
        point=point,
        library=library,
        target=source.target,
        roles=None,
    )


def _reset_seed_capture(
    *,
    missing: str | None = None,
    value: YamlValue = None,
    absent: bool = False,
) -> ParameterContext:
    source = _reset_seed_source()
    if missing is not None:
        point = source.document()
        component = QubitSchema.model_validate(point.components["Q1"].model_dump())
        if absent:
            component.ext.pop(missing)
        else:
            component.ext[missing] = value
        point.components["Q1"] = component
        source = ParameterContext(
            point=point,
            library=source.library,
            target=source.target,
            roles=None,
        )
    return source.bind_aliases(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator")), {}
    )


def _reset_seed_projection(name: str, capture: ParameterContext) -> dict[str, object]:
    adapter = _registry().create(_RESET_READERS[name])
    schema = adapter.make_default_cfg(
        SessionEnv(parameters=capture, soc=None, soccfg=None)
    )
    raw = schema_to_raw_dict(schema, capture)
    paths = {
        "single-freq": ("sweep.freq.start", "sweep.freq.stop", "sweep.freq.expts"),
        "single-length": ("modules.tested_reset.pulse_cfg.freq",),
        "dual-freq": (
            "modules.tested_reset.pulse1_cfg.gain",
            "modules.tested_reset.pulse2_cfg.gain",
            "sweep.freq1.start",
            "sweep.freq1.stop",
            "sweep.freq1.expts",
            "sweep.freq2.start",
            "sweep.freq2.stop",
            "sweep.freq2.expts",
        ),
        "dual-length": (
            "modules.tested_reset.pulse1_cfg.freq",
            "modules.tested_reset.pulse2_cfg.freq",
            "modules.tested_reset.pulse1_cfg.gain",
            "modules.tested_reset.pulse2_cfg.gain",
        ),
        "dual-power": (
            "modules.tested_reset.pulse1_cfg.freq",
            "modules.tested_reset.pulse2_cfg.freq",
        ),
        "bath-length": (
            "modules.tested_reset.cavity_tone_cfg.freq",
            "modules.tested_reset.cavity_tone_cfg.gain",
        ),
        "bath-phase": (
            "modules.tested_reset.cavity_tone_cfg.freq",
            "modules.tested_reset.cavity_tone_cfg.gain",
        ),
    }
    result: dict[str, object] = {}
    for path in paths[name]:
        current: object = raw
        for part in path.split("."):
            if not isinstance(current, dict):
                raise AssertionError(f"Missing cfg section for {path}")
            current = current[part]
        result[path] = current
    return result


def _reset_seed_expected(name: str, index: int) -> dict[str, object]:
    single = 4200.0 if index == 1 else 5200.0
    first, second = (4300.0, 4400.0) if index == 1 else (5300.0, 5400.0)
    gain1, gain2 = (0.21, 0.31) if index == 1 else (0.41, 0.51)
    bath_freq, bath_gain = (6300.0, 0.25) if index == 1 else (6400.0, 0.35)
    match name:
        case "single-freq":
            return {
                "sweep.freq.start": single - 12,
                "sweep.freq.stop": single + 12,
                "sweep.freq.expts": 201,
            }
        case "single-length":
            return {"modules.tested_reset.pulse_cfg.freq": single}
        case "dual-freq":
            return {
                "modules.tested_reset.pulse1_cfg.gain": gain1,
                "modules.tested_reset.pulse2_cfg.gain": gain2,
                "sweep.freq1.start": first - 50,
                "sweep.freq1.stop": first + 50,
                "sweep.freq1.expts": 201,
                "sweep.freq2.start": second - 50,
                "sweep.freq2.stop": second + 50,
                "sweep.freq2.expts": 201,
            }
        case "dual-length":
            return {
                "modules.tested_reset.pulse1_cfg.freq": first,
                "modules.tested_reset.pulse2_cfg.freq": second,
                "modules.tested_reset.pulse1_cfg.gain": gain1,
                "modules.tested_reset.pulse2_cfg.gain": gain2,
            }
        case "dual-power":
            return {
                "modules.tested_reset.pulse1_cfg.freq": first,
                "modules.tested_reset.pulse2_cfg.freq": second,
            }
        case "bath-length" | "bath-phase":
            return {
                "modules.tested_reset.cavity_tone_cfg.freq": bath_freq,
                "modules.tested_reset.cavity_tone_cfg.gain": bath_gain,
            }
        case _:
            raise ValueError(name)


@pytest.mark.parametrize("name", list(_RESET_READERS), ids=list(_RESET_READERS))
def test_reset_defaults_lower_accumulated_extensions_on_fixed_capture(
    name: str,
) -> None:
    capture = _reset_seed_capture()
    assert _reset_seed_projection(name, capture) == pytest.approx(
        _reset_seed_expected(name, 1)
    )


@pytest.mark.parametrize(
    ("name", "field", "state"),
    [
        (name, field, state)
        for name, field in _RESET_REQUIRED
        for state in ("absent", "null")
    ],
    ids=[
        f"{name}-{field}-{state}"
        for name, field in _RESET_REQUIRED
        for state in ("absent", "null")
    ],
)
def test_reset_defaults_reject_missing_accumulated_parameter(
    name: str, field: str, state: str
) -> None:
    capture = _reset_seed_capture(missing=field, absent=state == "absent")
    with pytest.raises(RuntimeError, match="Config field") as error:
        _reset_seed_projection(name, capture)
    assert f"$qubit.ext.{field}" in str(error.value)


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        (name, invalid)
        for name in _RESET_READERS
        for invalid in ("bool", "string", "nonfinite")
    ],
    ids=[
        f"{name}-{invalid}"
        for name in _RESET_READERS
        for invalid in ("bool", "string", "nonfinite")
    ],
)
def test_reset_defaults_reject_invalid_accumulated_parameter(
    name: str, invalid: str
) -> None:
    fields = {
        "single-freq": "reset_f",
        "single-length": "reset_f",
        "dual-freq": "reset_f1",
        "dual-length": "reset_f1",
        "dual-power": "reset_f1",
        "bath-length": "reset_bath_freq",
        "bath-phase": "reset_bath_freq",
    }
    values: dict[str, YamlValue] = {
        "bool": True,
        "string": "not-a-number",
        "nonfinite": float("inf"),
    }
    capture = _reset_seed_capture(missing=fields[name], value=values[invalid])
    with pytest.raises(RuntimeError, match="Config field") as error:
        _reset_seed_projection(name, capture)
    assert f"$qubit.ext.{fields[name]}" in str(error.value)


@pytest.mark.io
@pytest.mark.parametrize("name", list(_RESET_READERS), ids=list(_RESET_READERS))
def test_reset_default_capture_isolated_from_later_point_and_aliases(
    tmp_path: Path, name: str
) -> None:
    entry = ResultEntry.create(
        "seed-entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    entry.new_point("first")
    entry.new_point("second")
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    context.install_entry(entry)
    control = ContextControlFacet(context=context)
    source = _reset_seed_source()
    document = source.document()
    for point in ("first", "second"):
        control.use_point(PointUseRequest(point))
        for component, model in document.components.items():
            control.add_component(
                component, model.kind, model.model_dump(exclude={"kind"})
            )
    control.use_point(PointUseRequest("first"))
    target = control.read_context().target
    for root in source.library.module_keys():
        control.set_module(target, root, source.library.module_spec(root))
    control.set_alias_rows(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator"))
    )
    first = control.read_context()
    first_values = _reset_seed_projection(name, first)
    control.use_point(PointUseRequest("second"))
    control.set_alias_rows(
        (AliasRow("qubit", "Q2"), AliasRow("resonator", "$qubit.resonator"))
    )
    second = control.read_context()
    assert (
        _reset_seed_projection(name, first)
        == first_values
        == pytest.approx(_reset_seed_expected(name, 1))
    )
    assert _reset_seed_projection(name, second) == pytest.approx(
        _reset_seed_expected(name, 2)
    )
    assert control.read_context().target == second.target
