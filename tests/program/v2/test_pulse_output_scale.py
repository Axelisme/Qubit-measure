"""Observe the pulse representation gain contract through QICK compilation."""

import numpy as np
import pytest
from qick import QickConfig
from zcu_tools.program import v2
from zcu_tools.simulate.boundary import PulseEvent

from tests.program.v2.simulation._compiled_support import program, request


def _i_area(event: PulseEvent) -> float:
    if event.envelope is None:
        return event.gain * event.duration_ticks
    return (
        event.gain
        * event.envelope.sample_period_ticks
        * sum(sample.real for sample in event.envelope.values)
    )


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

    prog, env = program(body, initialize, reps=1, soccfg=mock_soccfg)
    dds, up, plateau, down, arb = request(prog, env).materialize_point(0).events
    # QICK truncates gain words, and the flat-top ramp word must be divisible
    # by the plateau factor's denominator. Allow four DAC gain LSBs per tick.
    tolerance = 4 * dds.duration_ticks / gen["maxv"]
    assert const_scale == 1.0
    assert _i_area(dds) < 0
    assert _i_area(arb) == pytest.approx(_i_area(dds), abs=tolerance)
    assert sum(map(_i_area, (up, plateau, down))) == pytest.approx(
        _i_area(dds), abs=tolerance
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
