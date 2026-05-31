from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

from typing_extensions import cast
from zcu_tools.experiment.v2_gui.adapters.shared import (
    make_readout_ref_default,
    schema_from_module,
    select_named_module_value,
)
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionValue,
    DirectValue,
    DisabledRefValue,
    ModuleRefValue,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import (
    AbsReadoutCfg,
    ModuleCfgFactory,
    PulseReadoutCfg,
    WaveformCfgFactory,
)


def test_schema_from_module_returns_none_for_none():
    assert schema_from_module(None) is None


def _pulse_readout(ml: ModuleLibrary, *, freq: float = 6000.0) -> PulseReadoutCfg:
    return cast(
        PulseReadoutCfg,
        ModuleCfgFactory.from_raw(
            {
                "type": "readout/pulse",
                "pulse_cfg": {
                    "waveform": {"style": "const", "length": 1.0},
                    "ch": 1,
                    "nqz": 2,
                    "freq": freq,
                    "gain": 0.2,
                },
                "ro_cfg": {
                    "ro_ch": 2,
                    "ro_freq": freq,
                    "ro_length": 1.0,
                    "trig_offset": 0.5,
                },
            },
            ml=ml,
        ),
    )


def test_schema_from_module_converts_proposed_readout_faithfully():
    """edit_schema is the spec+value view of the proposed module (freq carried).

    The proposed module is built the way adapters now build it: a direct
    with_updates on the strongly-typed PulseReadoutCfg.
    """
    readout = _pulse_readout(ModuleLibrary())
    proposed = readout.with_updates(
        pulse_cfg=readout.pulse_cfg.with_updates(freq=6150.0),
        ro_cfg=readout.ro_cfg.with_updates(ro_freq=6150.0),
    )

    schema = schema_from_module(proposed)

    assert isinstance(schema, CfgSchema)
    pulse_cfg = schema.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert pulse_cfg.fields["freq"].value == 6150.0  # type: ignore[union-attr]


def test_schema_from_module_converts_proposed_waveform_faithfully():
    """A proposed waveform cfg auto-routes through the waveform converter."""
    readout = _pulse_readout(ModuleLibrary())
    proposed = readout.pulse_cfg.waveform.with_updates(length=4.5)

    schema = schema_from_module(proposed)

    assert isinstance(schema, CfgSchema)
    length = schema.value.fields["length"]
    assert length.value == 4.5  # type: ignore[union-attr]


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


def test_select_named_module_value_returns_none_when_preferred_missing():
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
        )
    )
    selected = select_named_module_value(
        ml=ml,
        module_type=AbsReadoutCfg,
        preferred_names=["readout_dpm", "readout_rf"],
    )
    assert selected is None


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


def test_make_readout_ref_default_prefers_readout_dpm_over_readout_rf():
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
        ),
        readout_dpm=ModuleCfgFactory.from_raw(
            {
                "type": "readout/pulse",
                "pulse_cfg": {
                    "waveform": {"style": "const", "length": 1.0},
                    "ch": 3,
                    "nqz": 2,
                    "freq": 6200.0,
                    "gain": 0.2,
                },
                "ro_cfg": {
                    "ro_ch": 4,
                    "ro_freq": 6200.0,
                    "ro_length": 1.0,
                    "trig_offset": 0.5,
                },
            },
            ml=ml,
        ),
    )

    module_ref = make_readout_ref_default(_make_ctx(ml))
    assert isinstance(module_ref, ModuleRefValue)
    assert module_ref.chosen_key == "readout_dpm"


def test_make_readout_ref_default_fallback_uses_directvalue_when_md_missing():
    ctx = _make_ctx(ModuleLibrary())
    module_ref = make_readout_ref_default(ctx)
    assert isinstance(module_ref, ModuleRefValue)
    readout = module_ref.value
    assert isinstance(readout, CfgSectionValue)

    pulse_cfg = readout.fields["pulse_cfg"]
    ro_cfg = readout.fields["ro_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert isinstance(ro_cfg, CfgSectionValue)

    pulse_freq = pulse_cfg.fields["freq"]
    ro_freq = ro_cfg.fields["ro_freq"]
    assert isinstance(pulse_freq, DirectValue)
    assert isinstance(ro_freq, DirectValue)
    assert pulse_freq.value == 6000.0
    assert ro_freq.value == 6000.0


def test_make_readout_ref_default_fallback_prefers_ro_waveform_if_present():
    from zcu_tools.gui.adapter import WaveformRefValue

    ml = ModuleLibrary()
    ml.register_waveform(
        ro_waveform=WaveformCfgFactory.from_raw(
            {"style": "const", "length": 1.7}, ml=ml
        )
    )
    module_ref = make_readout_ref_default(_make_ctx(ml))
    assert isinstance(module_ref, ModuleRefValue)
    readout = module_ref.value
    pulse_cfg = readout.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    waveform = pulse_cfg.fields["waveform"]
    assert isinstance(waveform, WaveformRefValue)
    assert waveform.chosen_key == "ro_waveform"


def test_make_readout_ref_default_returns_disabled_when_optional_and_empty():
    # ADR-0012: optional ref with an empty library → DisabledRefValue marker.
    ctx = _make_ctx(ModuleLibrary())
    module_ref = make_readout_ref_default(ctx, optional=True)

    assert isinstance(module_ref, DisabledRefValue)
