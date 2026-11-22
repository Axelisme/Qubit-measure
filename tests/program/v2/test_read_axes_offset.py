"""Native IQ offset opt-in at the compiled shot-sums seam."""

import numpy as np
import pytest
from zcu_tools.program.v2 import IQ, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import ConstWaveformCfg, PulseReadout

from tests.program._profiles import load_soccfg
from tests.program._shot_reduction import reduce_round
from tests.program._timeline import compile_modules
from tests.program.v2._mux_support import AcquisitionSoc, mux_soccfg, native_mux


def pulse_readout(name: str = "readout", *, channel: int = 0) -> PulseReadout:
    return PulseReadout.Cfg(
        gen_ch=channel,
        ro_ch=channel,
        nqz=1,
        freq=100.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=1.0,
    ).build(name)


def test_pulse_iq_opt_in_corrects_both_components_before_accumulating() -> None:
    soccfg = load_soccfg()
    soccfg["readouts"][0]["iq_offset"] = 4.0
    program = compile_modules(soccfg, [pulse_readout()])
    raw_iq = {0: np.array([[[[5.0, 6.0]], [[7.0, 2.0]]]])}
    corrected = shot_sums(program, (IQ("readout", offset_corrected=True),))
    uncorrected = shot_sums(program, (IQ("readout"),))
    # With a scalar 4.0 offset on both components, the corrected shots are
    # (1, 2) and (3, -2). The same input remains unmodified for the default axis.
    np.testing.assert_array_equal(
        reduce_round(corrected, raw_iq), [[2, 4, 0, 10, -4, 8]]
    )
    np.testing.assert_array_equal(
        reduce_round(uncorrected, raw_iq), [[2, 12, 8, 74, 44, 40]]
    )


@pytest.mark.parametrize(
    ("frequency_kind", "raw_shots", "raw_moments"),
    [
        ("channel-center", [[9, 10], [11, 6]], [2, 20, 16, 202, 156, 136]),
        ("away-from-center", [[5, 6], [7, 2]], [2, 12, 8, 74, 44, 40]),
    ],
)
def test_single_tone_mux_uses_compiled_pfb_offset(
    monkeypatch: pytest.MonkeyPatch,
    frequency_kind: str,
    raw_shots: list[list[int]],
    raw_moments: list[int],
) -> None:
    soccfg = mux_soccfg()
    adc = soccfg["readouts"][4]
    adc["iq_offset"] = 4.0
    cfg = native_mux((4,), mixer_freq=0.0)
    # The fixture's center at DC subtracts input and output offsets; away
    # from the center, only the output offset is at DC.
    cfg.tones[0].freq = 0.0 if frequency_kind == "channel-center" else adc["f_dds"] / 8
    program = compile_modules(soccfg, [cfg.build("readout")], reps=2)
    soc = AcquisitionSoc(soccfg.get_cfg())
    length = program.get_time_axis(0, length_only=True)
    assert isinstance(length, int)
    raw = np.array(raw_shots, dtype=np.int64) * length
    monkeypatch.setattr(soc, "poll_data", lambda: [(2, ([raw], {}))])
    reducer = shot_sums(program, (IQ("readout", offset_corrected=True),))
    corrected = program.acquire_summed(soc, reducer=reducer, progress=False)
    np.testing.assert_array_equal(corrected, [2, 4, 0, 10, -4, 8])
    default = shot_sums(program, (IQ("readout"),))
    np.testing.assert_array_equal(
        program.acquire_summed(soc, reducer=default, progress=False), raw_moments
    )


@pytest.mark.parametrize("order", [2, 3])
def test_corrected_iq_rejects_cubic_or_extra_moment_axes(order: int) -> None:
    program = compile_modules(load_soccfg(), [pulse_readout()])
    axes = (
        (IQ("readout", order=2, offset_corrected=True), IQ("readout"))
        if order == 2
        else (IQ("readout", order=3, offset_corrected=True),)
    )
    with pytest.raises(ValueError, match="one quadratic IQ axis"):
        shot_sums(program, axes)


def test_corrected_iq_rejects_multi_tone_even_with_a_tone_selected() -> None:
    program = compile_modules(mux_soccfg(), [native_mux((4, 1)).build("readout")])
    with pytest.raises(ValueError, match="single ADC"):
        shot_sums(program, (IQ("readout", tone=0, offset_corrected=True),))


def test_corrected_iq_rejects_multiple_integrated_reads_on_one_adc() -> None:
    builder = TimelineBuilder(load_soccfg())
    builder.emit(pulse_readout())
    builder.timeline_inc(2.0)
    builder.emit(pulse_readout("other"))
    program = builder.build()
    with pytest.raises(ValueError, match="one integrated read per shot"):
        shot_sums(program, (IQ("readout", offset_corrected=True),))


def test_corrected_iq_rejects_edge_counting() -> None:
    program = compile_modules(load_soccfg(), [pulse_readout()])
    # Other transport contracts likewise set the edge-counting declaration on
    # the built fixture because PulseReadout does not expose that native option.
    program.ro_chs[0]["edge_counting"] = True
    with pytest.raises(ValueError, match="edge counting"):
        shot_sums(program, (IQ("readout", offset_corrected=True),))


def test_corrected_iq_rejects_registered_shot_masks() -> None:
    builder = TimelineBuilder(load_soccfg(), reps=2)
    readout = pulse_readout()
    builder.emit(readout, tag="keep")
    builder.timeline_inc(2.0)
    builder.emit(readout, tag="data")
    builder.register_shot_mask_source({0: [0]}, lambda iq: iq[0][..., 0, 0] > 0)
    with pytest.raises(ValueError, match="shot masks"):
        shot_sums(builder.build(), (IQ("data", offset_corrected=True),))
