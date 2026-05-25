from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

from zcu_tools.experiment.v2_gui.adapters.shared import (
    build_readout_for_frequency,
    build_waveform_for_length,
    make_pulse_readout_edit_template,
    make_readout_ref_default,
    select_named_module_value,
    update_readout_value_frequency,
)
from zcu_tools.gui.adapter import (
    CfgSectionValue,
    DirectValue,
    ModuleRefValue,
    ScalarValue,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import AbsReadoutCfg, ModuleCfgFactory


def test_update_readout_value_frequency_updates_all_supported_paths():
    readout_value = CfgSectionValue(
        fields={
            "ro_freq": DirectValue(6000.0),
            "pulse_cfg": CfgSectionValue(
                fields={"freq": DirectValue(6000.0)},
            ),
            "ro_cfg": CfgSectionValue(
                fields={"ro_freq": DirectValue(6000.0)},
            ),
        }
    )

    updated = update_readout_value_frequency(readout_value, 6100.0)

    assert updated.fields["ro_freq"].value == 6100.0  # type: ignore[union-attr]
    assert updated.fields["pulse_cfg"].fields["freq"].value == 6100.0  # type: ignore[union-attr]
    assert updated.fields["ro_cfg"].fields["ro_freq"].value == 6100.0  # type: ignore[union-attr]


def test_make_pulse_readout_edit_template_uses_requested_defaults():
    schema = make_pulse_readout_edit_template(
        pulse_ch=3,
        pulse_freq=6200.0,
        ro_ch=7,
    )
    value = schema.value
    pulse_cfg = value.fields["pulse_cfg"]
    ro_cfg = value.fields["ro_cfg"]

    assert isinstance(pulse_cfg, CfgSectionValue)
    assert isinstance(ro_cfg, CfgSectionValue)
    assert pulse_cfg.fields["freq"].value == 6200.0  # type: ignore[union-attr]
    assert ro_cfg.fields["ro_freq"].value == 6200.0  # type: ignore[union-attr]
    assert pulse_cfg.fields["ch"].value == 3  # type: ignore[union-attr]
    assert ro_cfg.fields["ro_ch"].value == 7  # type: ignore[union-attr]


def test_build_readout_for_frequency_updates_existing_module():
    ml = ModuleLibrary()
    readout = ModuleCfgFactory.from_raw(
        {
            "type": "readout/pulse",
            "pulse_cfg": {
                "waveform": {"style": "const", "length": 1.0},
                "ch": 1,
                "nqz": 2,
                "freq": 6000.0,
                "gain": 0.2,
            },
            "ro_cfg": {
                "ro_ch": 2,
                "ro_freq": 6000.0,
                "ro_length": 1.0,
                "trig_offset": 0.5,
            },
        },
        ml=ml,
    )

    updated = build_readout_for_frequency(
        readout,
        freq=6150.0,
        pulse_ch=1,
        ro_ch=2,
        ml=ml,
    )

    assert updated is not None
    assert getattr(getattr(updated, "pulse_cfg"), "freq") == 6150.0
    assert getattr(getattr(updated, "ro_cfg"), "ro_freq") == 6150.0


def test_build_waveform_for_length_falls_back_to_flat_top():
    waveform = build_waveform_for_length(
        None,
        length=4.5,
        ml=ModuleLibrary(),
    )

    assert waveform is not None
    assert getattr(waveform, "length") == 4.5


def test_build_waveform_for_length_updates_existing_waveform():
    ml = ModuleLibrary()
    readout = ModuleCfgFactory.from_raw(
        {
            "type": "readout/pulse",
            "pulse_cfg": {
                "waveform": {
                    "style": "flat_top",
                    "length": 1.5,
                    "raise_waveform": {"style": "cosine", "length": 0.1},
                },
                "ch": 1,
                "nqz": 2,
                "freq": 6000.0,
                "gain": 0.2,
            },
            "ro_cfg": {
                "ro_ch": 2,
                "ro_freq": 6000.0,
                "ro_length": 1.0,
                "trig_offset": 0.5,
            },
        },
        ml=ml,
    )

    updated = build_waveform_for_length(readout, length=3.25, ml=ml)

    assert updated is not None
    assert getattr(updated, "length") == 3.25


def test_select_named_module_value_prefers_requested_name():
    ml = ModuleLibrary()
    ml.register_module(
        alt=ModuleCfgFactory.from_raw(
            {
                "type": "readout/direct",
                "ro_ch": 1,
                "ro_freq": 6000.0,
                "ro_length": 1.0,
                "trig_offset": 0.5,
            },
            ml=ml,
        ),
        readout_rf=ModuleCfgFactory.from_raw(
            {
                "type": "readout/direct",
                "ro_ch": 2,
                "ro_freq": 6100.0,
                "ro_length": 1.1,
                "trig_offset": 0.6,
            },
            ml=ml,
        ),
    )

    selected = select_named_module_value(
        ml=ml,
        module_type=AbsReadoutCfg,
        preferred_names=["missing", "readout_rf"],
    )

    assert selected is not None
    assert selected.name == "readout_rf"


def _make_ctx(ml: ModuleLibrary) -> "ExpContext":
    from zcu_tools.gui.adapter import ExpContext

    return ExpContext(
        md=MetaDict(),
        ml=ml,
        soc=None,
        soccfg=None,
    )


def test_make_readout_ref_default_uses_library_selection_before_fallback():
    ml = ModuleLibrary()
    ml.register_module(
        readout_rf=ModuleCfgFactory.from_raw(
            {
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
            },
            ml=ml,
        )
    )

    ctx = _make_ctx(ml)
    module_ref = make_readout_ref_default(ctx)

    assert isinstance(module_ref, ModuleRefValue)
    assert module_ref.chosen_key == "readout_rf"


def test_make_readout_ref_default_falls_back_to_custom_when_lib_empty():
    ctx = _make_ctx(ModuleLibrary())
    module_ref = make_readout_ref_default(ctx)

    assert isinstance(module_ref, ModuleRefValue)
    assert module_ref.chosen_key == "<Custom:Pulse Readout>"


def test_make_readout_ref_default_returns_none_when_optional_and_empty():
    ctx = _make_ctx(ModuleLibrary())
    module_ref = make_readout_ref_default(ctx, optional=True)

    assert module_ref is None
