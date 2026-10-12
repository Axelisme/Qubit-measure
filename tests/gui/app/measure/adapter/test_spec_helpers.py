"""Proposed module and waveform conversion through the authoring seam."""

from zcu_tools.gui.app.measure.spec_helpers import schema_from_module
from zcu_tools.gui.cfg import CfgSchema, CfgSectionValue, DirectValue
from zcu_tools.program.v2 import ModuleCfgFactory, PulseReadoutCfg


def test_schema_from_module_returns_none_for_none():
    assert schema_from_module(None) is None


def _pulse_readout(*, freq: float = 6000.0) -> PulseReadoutCfg:
    cfg = ModuleCfgFactory.from_raw(
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
    )
    assert isinstance(cfg, PulseReadoutCfg)
    return cfg


def test_schema_from_module_converts_proposed_readout_faithfully():
    """edit_schema is the spec+value view of the proposed module (freq carried).

    The proposed module is built the way adapters now build it: a direct
    with_updates on the strongly-typed PulseReadoutCfg.
    """
    readout = _pulse_readout()
    proposed = readout.with_updates(
        pulse_cfg=readout.pulse_cfg.with_updates(freq=6150.0),
        ro_cfg=readout.ro_cfg.with_updates(ro_freq=6150.0),
    )

    schema = schema_from_module(proposed)

    assert isinstance(schema, CfgSchema)
    pulse_cfg = schema.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    freq = pulse_cfg.fields["freq"]
    assert isinstance(freq, DirectValue)
    assert freq.value == 6150.0


def test_schema_from_module_converts_proposed_waveform_faithfully():
    """A proposed waveform cfg auto-routes through the waveform converter."""
    readout = _pulse_readout()
    proposed = readout.pulse_cfg.waveform.with_updates(length=4.5)

    schema = schema_from_module(proposed)

    assert isinstance(schema, CfgSchema)
    length = schema.value.fields["length"]
    assert isinstance(length, DirectValue)
    assert length.value == 4.5
