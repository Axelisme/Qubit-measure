from __future__ import annotations

import pytest
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    PulseCfg,
    TableLengthPulse,
)
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)


def _pulse(waveform) -> PulseCfg:
    return PulseCfg(
        waveform=waveform,
        ch=0,
        nqz=1,
        freq=4000.0,
        gain=0.1,
    )


def _program(module: TableLengthPulse, count: int) -> ModularProgramV2:
    return ModularProgramV2(
        make_mock_soccfg(n_gens=1, n_readouts=1),
        ProgramV2Cfg(),
        modules=[module],
        sweep=[("length", count)],
    )


def test_const_uses_one_template_and_runtime_length_table() -> None:
    lengths = [0.1, 0.2, 0.4]
    module = TableLengthPulse(
        "probe",
        _pulse(ConstWaveformCfg(length=1.0)),
        lengths=lengths,
        idx_reg="length",
    )

    prog = _program(module, len(lengths))

    assert prog.pulse_registry.count == 1
    assert len(prog.pulses) == 1
    assert "w4" in prog.asm()
    assert "WPORT_WR" in prog.asm()
    dmem = prog.compile_datamem()
    assert dmem is not None
    assert len(dmem) == 2 * len(lengths)


def test_flat_top_uses_one_template_and_patches_only_flat_segment() -> None:
    lengths = [0.2, 0.3, 0.6]
    module = TableLengthPulse(
        "probe",
        _pulse(
            FlatTopWaveformCfg(
                length=1.0,
                raise_waveform=GaussWaveformCfg(length=0.1, sigma=0.025),
            )
        ),
        lengths=lengths,
        idx_reg="length",
    )

    prog = _program(module, len(lengths))

    assert prog.pulse_registry.count == 1
    (pulse,) = prog.pulses.values()
    assert len(pulse.get_wavenames(exclude_special=True)) == 3
    assert prog.asm().count("WPORT_WR") >= 3
    assert "w4" in prog.asm()


def test_flat_top_preserves_int4_dummy_segment() -> None:
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    soccfg["gens"][0]["type"] = "axis_sg_int4_v1"
    module = TableLengthPulse(
        "probe",
        _pulse(
            FlatTopWaveformCfg(
                length=1.0,
                raise_waveform=GaussWaveformCfg(length=0.1, sigma=0.025),
            )
        ),
        lengths=[0.2, 0.3, 0.6],
        idx_reg="length",
    )

    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(),
        modules=[module],
        sweep=[("length", 3)],
    )

    (pulse,) = prog.pulses.values()
    assert len(pulse.get_wavenames()) == 4
    assert prog.asm().count("WPORT_WR") >= 4


def test_rejects_unsupported_waveform() -> None:
    with pytest.raises(NotImplementedError, match="const and flat_top"):
        TableLengthPulse(
            "probe",
            _pulse(GaussWaveformCfg(length=0.1, sigma=0.025)),
            lengths=[0.1, 0.2],
            idx_reg="length",
        )


def test_rejects_zero_length_before_program_build() -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        TableLengthPulse(
            "probe",
            _pulse(ConstWaveformCfg(length=1.0)),
            lengths=[0.0, 0.1],
            idx_reg="length",
        )


def test_rejects_nonzero_length_below_hardware_minimum() -> None:
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    too_short = float(soccfg.cycles2us(2, gen_ch=0))
    module = TableLengthPulse(
        "probe",
        _pulse(ConstWaveformCfg(length=1.0)),
        lengths=[too_short, 0.1],
        idx_reg="length",
    )

    with pytest.raises(ValueError, match="requires at least 3"):
        ModularProgramV2(
            soccfg,
            ProgramV2Cfg(),
            modules=[module],
            sweep=[("length", 2)],
        )


def test_flat_top_nonzero_length_must_exceed_ramp() -> None:
    module = TableLengthPulse(
        "probe",
        _pulse(
            FlatTopWaveformCfg(
                length=1.0,
                raise_waveform=GaussWaveformCfg(length=0.1, sigma=0.025),
            )
        ),
        lengths=[0.05, 0.2],
        idx_reg="length",
    )

    with pytest.raises(ValueError, match="must exceed the fixed ramp"):
        _program(module, 2)
