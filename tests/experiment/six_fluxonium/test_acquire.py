"""Six independent physical cavities through the normal compiled acquire seam."""

from contextlib import contextmanager

import numpy as np
import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.experiment.six_fluxonium import six_fluxonium
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.dmem import LoadWord
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import (
    MuxReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.utils import readout_freq_words
from zcu_tools.resources.six_fluxonium import parse_six_fluxonium

from tests.program.v2.simulation.support import request_of
from tests.resources._six_fluxonium_data import six_document

pytestmark = pytest.mark.sim
FREQUENCIES = np.array([6800, 6925, 7050, 7175, 7300, 7425])


@contextmanager
def environment(document=None):
    try:
        env = six_fluxonium(
            parse_six_fluxonium(six_document() if document is None else document)
        )
    except NotImplementedError:
        pytest.fail(
            "six independent physical models must assemble a working acquire environment"
        )
    try:
        yield env
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def mux_program(
    env, *, order=(4, 1, 5, 0, 3, 2), cavities=(2, 5, 0, 4, 1, 3), drive=None
):
    readout = MuxReadoutCfg(
        gen_ch=0,
        nqz=1,
        mixer_freq=7100,
        length=0.064,
        ro_length=0.026,
        tones=[
            MuxReadoutTone(
                ro_ch=ch, freq=float(FREQUENCIES[cavity]), gain=0.04 + 0.01 * ch
            )
            for ch, cavity in zip(order, cavities, strict=True)
        ],
    )
    modules = [] if drive is None else [drive.build("drive")]
    return ModularProgramV2(
        env.soccfg, ProgramV2Cfg(reps=256), modules=[*modules, readout.build("mux")]
    )


def pulse_program(env, frequencies):
    gen, adc = readout_freq_words(
        env.soccfg, frequencies, gen_ch=13, ro_ch=6, mixer_freq=None, nqz=1
    )
    readout = PulseReadoutCfg(
        gen_ch=13,
        ro_ch=6,
        nqz=1,
        freq=float(frequencies[0]),
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.064),
        ro_length=0.026,
    )
    return ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=256),
        modules=[
            LoadWord("gen", gen, "axis", "gen_word"),
            LoadWord("adc", adc, "axis", "adc_word"),
            PulseReadout("pulse", readout, freq_val="gen_word", ro_freq_val="adc_word"),
        ],
        sweep=[("axis", len(frequencies))],
    )


def scalar_cavity_means(env, channel, cavity_indices):
    """Closed scalar vacuum-cavity response, independent of the ADC solver."""
    request = request_of(env)
    result = np.empty((len(cavity_indices), 2))
    for point, cavity in enumerate(cavity_indices):
        timeline = request.materialize_point(point)
        window = next(w for w in timeline.readouts if w.channel == channel)
        tone = next(
            e
            for e in timeline.events
            if e.gen_ch in (0, 13)
            and abs(e.frequency_mhz - window.frequency_mhz[0]) < 0.001
        )
        times = (window.trigger_us[0] - tone.start_us) * 1000 + np.arange(
            window.samples
        ) * 1000 / window.sample_rate_mhz
        pole = (
            np.pi * 0.001
            + 2j * np.pi * (FREQUENCIES[cavity] - tone.frequency_mhz) / 1000
        )
        # External kappa/(2*pi)=1 MHz, no internal loss, field calibration=10.
        emitted = (
            10
            * tone.gain
            * (1 - np.pi * 0.002 / (2 * pole) * (1 - np.exp(-pole * times)))
        )
        phase = np.exp(1j * np.deg2rad(window.phase_deg[0] - tone.phase_deg))
        expected = 10000 * np.conj(emitted).sum() * phase / window.samples
        result[point] = [expected.real, expected.imag]
    return result


def test_six_tones_follow_frequency_not_adc_or_declaration_order_and_keep_raw():
    with environment() as env:
        order, cavities = (4, 1, 5, 0, 3, 2), (2, 5, 0, 4, 1, 3)
        program = mux_program(env, order=order, cavities=cavities)
        average = program.acquire(env.soc, progress=False)
        assert average is not None and len(average) == 6
        for result, channel, cavity in zip(average, order, cavities, strict=True):
            np.testing.assert_allclose(
                np.asarray(result).reshape(-1, 2),
                scalar_cavity_means(env, channel, [cavity]),
                atol=0.03,
                rtol=1e-5,
            )
        request = request_of(env)
        # First materialization occurs after another acquisition and backend close.
        mux_program(env).acquire(env.soc, progress=False)
        cleanup = env.close_backend
        if cleanup is None:
            raise AssertionError("six-cavity environment must expose backend cleanup")
        cleanup()
        raw = program.get_raw()
        assert raw is not None and len(raw) == 6
        for value, result, window in zip(raw, average, request.readouts, strict=True):
            assert value.shape == (256, 1, 2) and value.dtype == np.int64
            # Raw rounds one integrated sum to int64, at most 0.5 ADC sums.
            # Divide by samples, then allow the direct mean's 0.03 ADC units.
            np.testing.assert_allclose(
                value.mean(axis=0) / window.samples,
                result,
                atol=0.5 / window.samples + 0.03,
                rtol=1e-5,
            )
        retained = program.get_raw()
        if retained is None:
            raise AssertionError("completed acquire must retain materialized shots")
        for value, expected in zip(retained, raw, strict=True):
            np.testing.assert_array_equal(value, expected)


def test_single_pulse_sweep_visits_all_six_cavities_per_point():
    frequencies = np.array([f + offset for f in FREQUENCIES for offset in (-10, 0, 10)])
    with environment() as env:
        program = pulse_program(env, frequencies)
        average = program.acquire(env.soc, progress=False)
        assert average is not None and len(average) == 1
        expected = scalar_cavity_means(env, 6, np.repeat(np.arange(6), 3))
        np.testing.assert_allclose(
            np.asarray(average[0]).reshape(-1, 2), expected, atol=0.03, rtol=1e-5
        )
        raw = program.get_raw()
        assert raw is not None and raw[0].shape == (256, 18, 1, 2)


def test_decimated_channels_keep_declared_order_and_companion_sums():
    with environment() as env:
        program = mux_program(env)
        traces = program.acquire_decimated(env.soc, progress=False)
        raw = program.get_raw()
        assert traces is not None and raw is not None and len(traces) == len(raw) == 6
        for channel, (trace, accumulated) in enumerate(zip(traces, raw, strict=True)):
            values = np.asarray(trace)
            assert values.shape[0] == 256, channel
            np.testing.assert_allclose(
                accumulated.reshape(256, 2),
                values.reshape(256, -1, 2).sum(axis=1),
                atol=(values.shape[-2] + 1) / 2,
                rtol=0,
            )


def test_drive_affects_only_its_own_cavity_with_global_dc_live_refresh():
    document = six_document()
    for q in document["qubits"].values():
        q["dc_bias"]["offset_phi0"] = 0.5
        q["readout"]["resonator"]["charge_coefficient_ghz"] = 0.03
    with environment(document) as env:
        baseline = mux_program(
            env, order=(0, 1, 2, 3, 4, 5), cavities=(0, 1, 2, 3, 4, 5)
        ).acquire(env.soc, progress=False)
        driven = mux_program(
            env,
            order=(0, 1, 2, 3, 4, 5),
            cavities=(0, 1, 2, 3, 4, 5),
            drive=PulseCfg(
                ch=3, nqz=1, freq=580, gain=0.2, waveform=ConstWaveformCfg(length=0.032)
            ),
        ).acquire(env.soc, progress=False)
        assert baseline is not None and driven is not None
        changes = np.array(
            [np.linalg.norm(a - b) for a, b in zip(baseline, driven, strict=True)]
        )
        assert changes[2] > 0.1
        assert np.max(changes[[0, 1, 3, 4, 5]]) < 0.1
        source = env.create_device("FakeDevice", "flux_bias")
        assert env.soc.binding.capture_snapshot().values == (("flux_bias", 0.0),)
        source.set_value(0.2)
        assert env.soc.binding.capture_snapshot().values == (("flux_bias", 0.2),)


@pytest.mark.parametrize("output", ["integrated", "deferred_raw", "decimated"])
def test_different_sweep_sizes_do_not_reuse_cavity_noise(output):
    document = six_document()
    # Pulse ADC6 uses Q1's receiver, even when its frequency selects cavity 1.
    document["qubits"]["Q1"]["readout"]["adc"]["noise_psd_adc2_ns"] = [
        [100, 0],
        [0, 100],
    ]
    with environment(document) as env:
        # Both acquires give cavity 1 a single-point partition starting at 0.
        # Extra points visit cavity 0, so only the total request size changes.
        long = pulse_program(env, [FREQUENCIES[1], *([FREQUENCIES[0]] * 12)])
        short = pulse_program(env, [FREQUENCIES[1]])
        if output == "decimated":
            first = long.acquire_decimated(env.soc, progress=False)
            second = short.acquire_decimated(env.soc, progress=False)
        else:
            first = long.acquire(env.soc, progress=False)
            second = short.acquire(env.soc, progress=False)

    # Raw materialization must keep its distinct identity after the next acquire
    # and backend close, not borrow a live RNG or the most recent request.
    if output == "deferred_raw":
        first, second = long.get_raw(), short.get_raw()
    assert first is not None and second is not None
    if output == "integrated":
        first_point = np.asarray(first[0]).reshape(13, -1, 2)[0]
        next_point = np.asarray(second[0]).reshape(1, -1, 2)[0]
    else:
        first_point = np.asarray(first[0]).reshape(256, 13, -1, 2)[:, 0]
        next_point = np.asarray(second[0]).reshape(256, 1, -1, 2)[:, 0]
    assert not np.array_equal(first_point, next_point), (
        "independent acquires must not reuse the same cavity receiver noise"
    )


def test_equidistant_readout_tone_fails_before_any_round_result():
    document = six_document()
    document["qubits"]["Q1"]["readout"]["resonator"]["frequency_ghz"] = 6.8
    document["qubits"]["Q2"]["readout"]["resonator"]["frequency_ghz"] = 7.024
    with environment(document) as env:
        program = pulse_program(env, [6912.0])
        with pytest.raises(ValueError, match="equidistant|tie|ambiguous"):
            program.acquire(env.soc, progress=False)
