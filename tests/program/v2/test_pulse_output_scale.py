"""Observe the pulse representation gain contract through QICK compilation."""

from typing import override

import numpy as np
import pytest
from numpy.typing import NDArray
from qick import QickConfig
from qick.asm_v2 import QickRawParam, Waveform
from zcu_tools.program import v2
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    CosineWaveformCfg,
    FlatTopWaveformCfg,
)


def _native_waves(prog: v2.TimelineProgramV2, name: str) -> list[Waveform]:
    waves: list[Waveform] = []
    for wave in prog.pulses[name].waveforms:
        assert isinstance(wave, Waveform)
        waves.append(wave)
    return waves


def _word(value: int | QickRawParam) -> int:
    return value.start if isinstance(value, QickRawParam) else value


def _i_area(
    wave: Waveform, samples_per_cycle: int, idata: NDArray[np.int16] | None = None
) -> float:
    gain, cycles = _word(wave.gain), _word(wave.length)
    if idata is None:
        return gain * cycles
    first = wave.env * samples_per_cycle
    last = first + cycles * samples_per_cycle
    return gain * float(idata[first:last].sum()) / (65536 * samples_per_cycle)


@pytest.mark.parametrize(
    ("custom_maxv", "rectangle_cycles"),
    [(None, 12), (24775, 11)],
    ids=["ordinary-generator", "reduced-maxv-with-headroom"],
)
def test_matched_gains_preserve_compiled_i_area(
    mock_soccfg: QickConfig, custom_maxv: int | None, rectangle_cycles: int
) -> None:
    gen = mock_soccfg["gens"][0]
    if custom_maxv is not None:
        # Gives a full-scale ramp at 3/4 of the DDS plateau in this fixture.
        gen.update(maxv=custom_maxv, maxv_scale=0.744)
    samples_per_cycle = gen["samps_per_clk"]
    peak = mock_soccfg.get_maxv(0)
    const_gain = -0.3
    const_scale = v2.pulse_output_scale(mock_soccfg, 0, "const")
    flat_gain = (
        const_gain * const_scale / v2.pulse_output_scale(mock_soccfg, 0, "flat_top")
    )
    arb_gain = const_gain * const_scale / v2.pulse_output_scale(mock_soccfg, 0, "arb")

    def initialize(prog: v2.TimelineProgramV2) -> None:
        prog.declare_gen(0, nqz=1)
        # Four full ramp cycles plus an eight-cycle plateau give 12 equivalent
        # cycles on the ordinary generator (within quantization) and 11 on the mock.
        ramp = np.repeat([0, 0, peak, peak, peak, peak, 0, 0], samples_per_cycle)
        prog.add_envelope(0, "ramp", idata=ramp)
        prog.add_envelope(
            0, "rectangle", idata=np.full(rectangle_cycles * samples_per_cycle, peak)
        )
        prog.add_pulse(
            0,
            "dds",
            style="const",
            freq=0,
            phase=0,
            gain=const_gain,
            length=prog.cycles2us(rectangle_cycles, gen_ch=0),
        )
        prog.add_pulse(
            0,
            "flat",
            style="flat_top",
            freq=0,
            phase=0,
            gain=flat_gain,
            length=prog.cycles2us(8, gen_ch=0),
            envelope="ramp",
        )
        prog.add_pulse(
            0,
            "arb",
            style="arb",
            freq=0,
            phase=0,
            gain=arb_gain,
            envelope="rectangle",
        )

    def body(prog: v2.TimelineProgramV2) -> None:
        for name in ("dds", "flat", "arb"):
            prog.pulse(0, name)

    class AreaProgram(v2.TimelineProgramV2):
        @override
        def _initialize(self, cfg):
            initialize(self)

        @override
        def _body(self, cfg):
            body(self)

    prog = AreaProgram(mock_soccfg, reps=1, initial_delay=0, relax_delay=0)
    (dds,) = _native_waves(prog, "dds")
    up, plateau, down = _native_waves(prog, "flat")
    (arb,) = _native_waves(prog, "arb")
    envelopes = prog.envelopes[0]["envs"]
    assert isinstance(envelopes, dict)
    idata = np.concatenate(
        [envelopes[name]["data"][:, 0] for name in ("ramp", "rectangle")]
    )
    dds_area = _i_area(dds, samples_per_cycle)
    flat_area = (
        _i_area(up, samples_per_cycle, idata)
        + _i_area(plateau, samples_per_cycle)
        + _i_area(down, samples_per_cycle, idata)
    )
    # QICK truncates gain words, and the flat-top ramp word must be divisible
    # by the plateau factor's denominator. Allow four DAC gain LSBs per cycle.
    tolerance = 4 * _word(dds.length)
    assert const_scale == 1.0
    assert dds_area < 0
    assert _i_area(arb, samples_per_cycle, idata) == pytest.approx(
        dds_area, abs=tolerance
    )
    assert flat_area == pytest.approx(dds_area, abs=tolerance)


def test_flat_top_plateau_scale_is_not_the_larger_ramp_peak(
    mock_soccfg: QickConfig,
) -> None:
    gen = mock_soccfg["gens"][0]
    gen.update(maxv=32766, maxv_scale=0.89)
    cycle_us = mock_soccfg.cycles2us(1, gen_ch=0)
    const = v2.Pulse(
        "const",
        v2.PulseCfg(
            ch=0,
            nqz=1,
            freq=0,
            gain=0.95,
            waveform=ConstWaveformCfg(length=8 * cycle_us),
        ),
    )
    flat = v2.Pulse(
        "flat",
        v2.PulseCfg(
            ch=0,
            nqz=1,
            freq=0,
            gain=0.95,
            waveform=FlatTopWaveformCfg(
                length=16 * cycle_us,
                raise_waveform=CosineWaveformCfg(length=8 * cycle_us),
            ),
        ),
    )
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.emit(const)
    builder.emit(flat)
    compiled = builder.build()
    dds = compiled.pulses[const.pulse_name(compiled)].waveforms[0]
    plateau = compiled.pulses[flat.pulse_name(compiled)].waveforms[1]
    assert isinstance(dds, Waveform) and isinstance(dds.gain, QickRawParam)
    assert isinstance(plateau, Waveform) and isinstance(plateau.gain, QickRawParam)

    plateau_scale = v2.flat_top_plateau_scale(mock_soccfg, 0)
    assert plateau_scale == pytest.approx(0.4993757802746567)
    assert plateau_scale == pytest.approx(plateau.gain.start / dds.gain.start, abs=2e-4)
    assert (
        plateau_scale
        < v2.pulse_output_scale(mock_soccfg, 0, "flat_top")
        == v2.pulse_output_scale(mock_soccfg, 0, "arb")
    )


@pytest.mark.parametrize("style", ["gauss"])
def test_pulse_output_scale_rejects_unknown_style(
    mock_soccfg: QickConfig, style
) -> None:
    with pytest.raises(ValueError, match="style.*gauss"):
        v2.pulse_output_scale(mock_soccfg, 0, style)


@pytest.mark.parametrize("gen_ch", [-1, 99])
def test_pulse_output_scale_rejects_missing_generator_channel(
    mock_soccfg: QickConfig, gen_ch: int
) -> None:
    with pytest.raises(ValueError, match=rf"gen_ch.*{gen_ch}"):
        v2.pulse_output_scale(mock_soccfg, gen_ch, "const")


def test_pulse_output_scale_rejects_missing_gens(mock_soccfg: QickConfig) -> None:
    soccfg = QickConfig({"sw_version": mock_soccfg["sw_version"]})
    with pytest.raises(ValueError, match="gens"):
        v2.pulse_output_scale(soccfg, 0, "const")


@pytest.mark.parametrize("field", ["maxv", "maxv_scale"])
def test_pulse_output_scale_rejects_missing_generator_field(
    mock_soccfg: QickConfig, field: str
) -> None:
    del mock_soccfg["gens"][0][field]
    with pytest.raises(ValueError, match=rf"generator 0.*{field}"):
        v2.pulse_output_scale(mock_soccfg, 0, "const")
