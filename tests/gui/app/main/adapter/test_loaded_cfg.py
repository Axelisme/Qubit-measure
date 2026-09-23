from copy import deepcopy

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.onetone.freq import FreqCfg, HomophasalSamplingCfg
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import OneToneFreqAdapter
from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.adapter.loaded_cfg import project_loaded_cfg
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.adapter.types import RunRequest
from zcu_tools.gui.app.main.specs import make_bath_reset_spec, make_pulse_spec
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    is_custom_reference_key,
    make_default_value,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import PulseCfg
from zcu_tools.program.v2.modules.reset import BathResetCfg


class Snapshot(ExpCfgModel):
    reps: object = 12
    sweep: dict[str, object] = {}
    modules: dict[str, object] = {}


def test_projection_replaces_expressions_but_keeps_missing_fields_detached():
    current = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "reps": ScalarSpec("Reps", int),
                "uniform": ScalarSpec("Uniform", bool),
            }
        ),
        value=CfgSectionValue(
            fields={"reps": EvalValue("old", 2), "uniform": DirectValue(True)}
        ),
    )
    before = deepcopy(current)
    result = project_loaded_cfg(current, Snapshot())
    assert result is not None
    assert result.value.fields == {
        "reps": DirectValue(12),
        "uniform": DirectValue(True),
    }
    assert current == before
    result.value.fields["uniform"] = DirectValue(False)
    assert current == before


@pytest.mark.parametrize("value", [None, "12", True, [], {}])
def test_incompatible_scalar_does_not_count_as_adopted(value):
    current = CfgSchema(
        spec=CfgSectionSpec(fields={"reps": ScalarSpec("Reps", int)}),
        value=CfgSectionValue(fields={"reps": DirectValue(2)}),
    )
    assert project_loaded_cfg(current, Snapshot(reps=value)) is None
    assert current.value.fields["reps"] == DirectValue(2)


def test_optional_scalar_can_adopt_explicit_null():
    current = CfgSchema(
        spec=CfgSectionSpec(fields={"reps": ScalarSpec("Reps", int, optional=True)}),
        value=CfgSectionValue(fields={"reps": DirectValue(2)}),
    )
    result = project_loaded_cfg(current, Snapshot(reps=None))
    assert result is not None
    assert result.value.fields["reps"] == DirectValue(None)
    assert current.value.fields["reps"] == DirectValue(2)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"start": 1.0, "stop": 3.0, "expts": 3, "step": 1.0}, SweepValue(1.0, 3.0, 3)),
        ({"start": 1.0, "stop": 1.0, "expts": 1, "step": 0.0}, SweepValue(1.0, 1.0, 1)),
        ({"start": 1.0, "stop": 3.0, "expts": 3, "step": 2.0}, None),
        ([1.0, 2.0, 4.0], None),
    ],
)
def test_sweep_requires_reproducible_runtime_sequence(raw, expected):
    current = CfgSchema(
        spec=CfgSectionSpec(
            fields={"sweep": CfgSectionSpec(fields={"freq": SweepSpec()})}
        ),
        value=CfgSectionValue(
            fields={"sweep": CfgSectionValue(fields={"freq": SweepValue(5, 6, 2)})}
        ),
    )
    result = project_loaded_cfg(current, Snapshot(sweep={"freq": raw}))
    if expected is None:
        assert result is None
    else:
        assert result is not None
        section = result.value.fields["sweep"]
        assert isinstance(section, CfgSectionValue)
        assert section.fields["freq"] == expected


def test_complete_pulse_becomes_custom_with_nested_waveform():
    spec = CfgSectionSpec(
        fields={
            "modules": CfgSectionSpec(
                fields={
                    "pulse": ReferenceSpec(kind="module", allowed=[make_pulse_spec()])
                }
            )
        }
    )
    current = CfgSchema(spec=spec, value=make_default_value(spec))
    pulse = PulseCfg.model_validate(
        {
            "ch": 0,
            "nqz": 1,
            "freq": 5000.0,
            "gain": 0.2,
            "waveform": {"style": "const", "length": 0.1},
        }
    )
    result = project_loaded_cfg(current, Snapshot(modules={"pulse": pulse}))
    assert result is not None
    modules = result.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    projected = modules.fields["pulse"]
    assert isinstance(projected, ReferenceValue)
    assert is_custom_reference_key(projected.chosen_key)
    assert projected.value.fields["freq"] == DirectValue(5000.0)
    waveform = projected.value.fields["waveform"]
    assert isinstance(waveform, ReferenceValue)
    assert is_custom_reference_key(waveform.chosen_key)


def test_device_label_inverse_requires_unique_name():
    spec = CfgSectionSpec(
        fields={"dev": CfgSectionSpec(fields={"flux_dev": ScalarSpec("Flux", str)})}
    )
    current = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={"dev": CfgSectionValue(fields={"flux_dev": DirectValue("old")})}
        ),
    )
    info = FakeDeviceInfo(address="fake", label="flux_dev")
    result = project_loaded_cfg(current, Snapshot(dev={"flux_yoko": info}))
    assert result is not None
    dev = result.value.fields["dev"]
    assert isinstance(dev, CfgSectionValue)
    assert dev.fields["flux_dev"] == DirectValue("flux_yoko")
    assert project_loaded_cfg(current, Snapshot(dev={"a": info, "b": info})) is None


def test_device_selector_rejects_names_absent_from_live_options():
    spec = CfgSectionSpec(
        fields={
            "dev": CfgSectionSpec(
                fields={
                    "jpa_rf_dev": ScalarSpec(
                        "JPA RF device", str, required=True, choices_source="devices"
                    )
                }
            )
        }
    )
    current = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={"dev": CfgSectionValue(fields={"jpa_rf_dev": DirectValue("old")})}
        ),
    )
    info = FakeDeviceInfo(address="fake", label="jpa_rf_dev")

    assert (
        project_loaded_cfg(
            current,
            Snapshot(dev={"stale": info}),
            provide_options=lambda source: ("old",) if source == "devices" else (),
        )
        is None
    )
    assert project_loaded_cfg(current, Snapshot(dev={"stale": info})) is None
    adopted = project_loaded_cfg(
        current,
        Snapshot(dev={"available": info}),
        provide_options=lambda source: ("old", "available"),
    )
    assert adopted is not None
    dev = adopted.value.fields["dev"]
    assert isinstance(dev, CfgSectionValue)
    assert dev.fields["jpa_rf_dev"] == DirectValue("available")
    original_dev = current.value.fields["dev"]
    assert isinstance(original_dev, CfgSectionValue)
    assert original_dev.fields["jpa_rf_dev"] == DirectValue("old")


def test_multi_axis_sweep_preserves_run_only_uniform():
    spec = CfgSectionSpec(
        fields={
            "sweep": CfgSectionSpec(
                fields={"length": SweepSpec(), "gain": SweepSpec()}
            ),
            "uniform": ScalarSpec("Uniform", bool),
        }
    )
    current = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={
                "sweep": CfgSectionValue(
                    fields={"length": SweepValue(3, 4, 2), "gain": SweepValue(4, 5, 2)}
                ),
                "uniform": DirectValue(False),
            }
        ),
    )
    result = project_loaded_cfg(
        current,
        Snapshot(
            sweep={
                "length": {"start": 0.0, "stop": 10.0, "expts": 11, "step": 1.0},
                "gain": {"start": 0.1, "stop": 0.5, "expts": 3, "step": 0.2},
            }
        ),
    )
    assert result is not None
    sweep = result.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    assert sweep.fields["length"] == SweepValue(0.0, 10.0, 11)
    assert sweep.fields["gain"] == SweepValue(0.1, 0.5, 3)
    assert result.value.fields["uniform"] == DirectValue(False)


def test_onetone_freq_runtime_mode_and_readout_are_restored():
    adapter = OneToneFreqAdapter()
    ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    current = adapter.make_default_cfg(ctx)
    raw = schema_to_raw_dict(current, ctx.md, ctx.ml)
    runtime = adapter.build_exp_cfg(
        raw, RunRequest(md=ctx.md, ml=ctx.ml, soc=None, soccfg=None)
    )
    snapshot = FreqCfg.model_validate(
        {
            **runtime.model_dump(),
            "sampling_mode": "homophasal",
            "homophasal": HomophasalSamplingCfg(r_f=6000.0, rf_w=10.0, theta0=0.1),
        }
    )
    result = project_loaded_cfg(current, snapshot)
    assert result is not None
    assert result.value.fields["sampling_mode"] == DirectValue("homophasal")
    modules = result.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ReferenceValue)
    assert is_custom_reference_key(readout.chosen_key)
    pulse = readout.value.fields["pulse_cfg"]
    assert isinstance(pulse, CfgSectionValue)
    waveform = pulse.fields["waveform"]
    assert isinstance(waveform, ReferenceValue)
    assert is_custom_reference_key(waveform.chosen_key)


def test_bath_reset_nested_references_are_custom():
    pulse = PulseCfg.model_validate(
        {
            "ch": 0,
            "nqz": 1,
            "freq": 5000.0,
            "gain": 0.2,
            "waveform": {"style": "const", "length": 0.1},
        }
    )
    bath = BathResetCfg(cavity_tone_cfg=pulse, qubit_tone_cfg=pulse, pi2_cfg=pulse)
    spec = CfgSectionSpec(
        fields={
            "modules": CfgSectionSpec(
                fields={
                    "tested_reset": ReferenceSpec(
                        kind="module", allowed=[make_bath_reset_spec()]
                    )
                }
            )
        }
    )
    current = CfgSchema(spec=spec, value=make_default_value(spec))
    result = project_loaded_cfg(current, Snapshot(modules={"tested_reset": bath}))
    assert result is not None
    modules = result.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested_reset = modules.fields["tested_reset"]
    assert isinstance(tested_reset, ReferenceValue)
    assert is_custom_reference_key(tested_reset.chosen_key)
    for name in ("cavity_tone_cfg", "qubit_tone_cfg", "pi2_cfg"):
        child = tested_reset.value.fields[name]
        assert isinstance(child, ReferenceValue)
        assert is_custom_reference_key(child.chosen_key)
        assert isinstance(child.value.fields["waveform"], ReferenceValue)


def test_null_in_nested_module_preserves_the_entire_reference():
    spec = CfgSectionSpec(
        fields={
            "reps": ScalarSpec("Reps", int),
            "modules": CfgSectionSpec(
                fields={
                    "pulse": ReferenceSpec(kind="module", allowed=[make_pulse_spec()])
                }
            ),
        }
    )
    current = CfgSchema(spec=spec, value=make_default_value(spec))
    before = deepcopy(current)
    result = project_loaded_cfg(
        current,
        Snapshot(
            modules={
                "pulse": {
                    "type": "pulse",
                    "ch": 0,
                    "nqz": 1,
                    "freq": None,
                    "gain": 0.2,
                    "waveform": {"style": "const", "length": 0.1},
                }
            }
        ),
    )
    assert result is not None
    assert result.value.fields["reps"] == DirectValue(12)
    assert result.value.fields["modules"] == before.value.fields["modules"]
    assert current == before


def test_incomplete_module_does_not_invent_defaults():
    spec = CfgSectionSpec(
        fields={
            "modules": CfgSectionSpec(
                fields={
                    "pulse": ReferenceSpec(kind="module", allowed=[make_pulse_spec()])
                }
            )
        }
    )
    current = CfgSchema(spec=spec, value=make_default_value(spec))
    assert (
        project_loaded_cfg(
            current, Snapshot(modules={"pulse": {"type": "pulse", "freq": 5000.0}})
        )
        is None
    )
