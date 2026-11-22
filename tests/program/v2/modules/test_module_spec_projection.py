from pathlib import Path

from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ArbWaveformCfg as ArbWaveformSpec
from zcu_tools.cfg_model import FlatTopWaveformSpec, PulseResetSpec, PulseSpec
from zcu_tools.program.v2.modules import (
    ArbWaveformCfg,
    FlatTopWaveformCfg,
    ModuleCfgFactory,
    PulseCfg,
    PulseResetCfg,
)


def test_captured_arb_pulse_projected_without_reading_asset(tmp_path: Path) -> None:
    pulse = PulseCfg(
        ch=2,
        nqz=2,
        freq=500.0,
        gain=-0.1,
        phase=30.0,
        waveform=ArbWaveformCfg(data="shape", path=str(tmp_path / "missing.npz")),
    )
    capture = pulse.model_dump_json()

    proposal = ModuleCfgFactory.to_spec(
        PulseCfg.model_validate_json(capture), field_updates=(("gain", -0.25),)
    )

    assert proposal == PulseSpec(
        ch=2,
        nqz=2,
        freq=500.0,
        gain=-0.25,
        phase=30.0,
        waveform=ArbWaveformSpec(data="shape"),
    )


def test_nested_arb_projection_replaces_sweep_without_mutating_source(
    tmp_path: Path,
) -> None:
    reset = PulseResetCfg(
        pulse_cfg=PulseCfg(
            ch=2,
            nqz=2,
            freq=QickParam(500.0, {"freq": 25.0}),
            gain=0.2,
            waveform=FlatTopWaveformCfg(
                length=0.1,
                raise_waveform=ArbWaveformCfg(
                    data="ramp", path=str(tmp_path / "missing.npz")
                ),
            ),
        )
    )
    before = reset.to_dict()

    proposal = ModuleCfgFactory.to_spec(
        reset, field_updates=(("pulse_cfg.freq", 510.0),)
    )

    assert proposal == PulseResetSpec(
        pulse_cfg=PulseSpec(
            ch=2,
            nqz=2,
            freq=510.0,
            gain=0.2,
            waveform=FlatTopWaveformSpec(
                length=0.1, raise_waveform=ArbWaveformSpec(data="ramp")
            ),
        )
    )
    assert reset.to_dict() == before
