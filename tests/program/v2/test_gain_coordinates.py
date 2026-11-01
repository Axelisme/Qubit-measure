from tests.program._timeline import compile_modules
"""Host gain coordinates retain their intended words through QICK compilation."""

import numpy as np
import pytest
from qick import QickConfig
from qick.helpers import to_int
from zcu_tools.experiment.v2.utils.round_zcu import (
    round_sweep_dict,
    round_zcu_gain,
    sweep2array,
)
from zcu_tools.program.v2 import (
    Pulse,
    PulseCfg,
    SweepCfg,
    make_mock_soccfg,
    sweep2param,
)
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def _compile_scalar_word(soccfg: QickConfig, gain: float) -> int:
    pulse = Pulse(
        "drive",
        PulseCfg(
            ch=0,
            nqz=1,
            freq=100.0,
            gain=gain,
            waveform=ConstWaveformCfg(length=0.1),
        ),
    )
    prog = compile_modules(soccfg, [pulse])
    actual = prog.get_pulse_param(pulse.pulse_id, "gain", as_array=False)
    factor = soccfg["gens"][0]["maxv"] * soccfg["gens"][0]["maxv_scale"]
    return round(actual * factor)


def _compile_sweep_words(
    soccfg: QickConfig, sweep: SweepCfg, scaler: float
) -> np.ndarray:
    pulse = Pulse(
        "drive",
        PulseCfg(
            ch=0,
            nqz=1,
            freq=100.0,
            gain=scaler * sweep2param("gain", sweep),
            waveform=ConstWaveformCfg(length=0.1),
        ),
    )
    prog = compile_modules(soccfg, [pulse], sweep=[("gain", sweep)])
    actual = prog.get_pulse_param(pulse.pulse_id, "gain", as_array=True)
    factor = soccfg["gens"][0]["maxv"] * soccfg["gens"][0]["maxv_scale"]
    return np.rint(actual * factor).astype(int)


@pytest.mark.parametrize("scaler", [1.0, 2.0, 0.3, -1.0, -0.3])
@pytest.mark.parametrize("sign", [1, -1])
def test_signed_gain_returns_nearest_coordinate_that_compiles(
    scaler: float, sign: int
) -> None:
    soccfg = make_mock_soccfg()
    soccfg["gens"][0]["maxv"] = 30000
    requested = sign * 59.5 / (scaler * 30000)
    target = sign * 59
    ideal = target / (scaler * 30000)

    actual = round_zcu_gain(requested, soccfg, 0, scaler=scaler)

    assert _compile_scalar_word(soccfg, scaler * actual) == target
    nearest = ideal
    if to_int(scaler * ideal, 30000, trunc=True) != target:
        nearest = np.nextafter(ideal, np.copysign(np.inf, ideal))
    assert actual == nearest


@pytest.mark.parametrize(
    ("start", "stop", "expected"),
    [
        (0.1, 0.4, [3276, 6552, 9828, 13104]),
        (0.4, 0.1, [13106, 9830, 6554, 3278]),
        (-0.4, -0.1, [-13106, -9830, -6554, -3278]),
        (-0.1, -0.4, [-3276, -6552, -9828, -13104]),
    ],
)
def test_regular_host_gain_grid_retains_start_and_step_words(
    start, stop, expected
) -> None:
    soccfg = make_mock_soccfg()
    sweep = SweepCfg(start=start, stop=stop, expts=4, step=(stop - start) / 3)
    actual = sweep2array(sweep, "gain", {"soccfg": soccfg, "gen_ch": 0})
    words = [_compile_scalar_word(soccfg, float(g)) for g in actual]
    np.testing.assert_array_equal(words, expected)


@pytest.mark.parametrize(
    ("start", "stop", "expected"),
    [
        (0.1, 0.4, [3276, 6552, 9828, 13104]),
        (0.4, 0.1, [13106, 9830, 6554, 3278]),
        (-0.4, -0.1, [-13106, -9830, -6554, -3278]),
        (-0.1, -0.4, [-3276, -6552, -9828, -13104]),
    ],
)
@pytest.mark.parametrize("scaler", [1.0, 2.0, -1.0])
def test_hardware_gain_sweep_compiles_to_host_word_grid(
    start, stop, expected, scaler
) -> None:
    soccfg = make_mock_soccfg()
    sweep = SweepCfg(
        start=start / scaler,
        stop=stop / scaler,
        expts=4,
        step=(stop - start) / 3 / scaler,
    )
    info = {"soccfg": soccfg, "gen_ch": 0, "scaler": scaler}
    rounded = round_sweep_dict(sweep, "gain", info)
    words = _compile_sweep_words(soccfg, rounded, scaler)
    np.testing.assert_array_equal(words, expected)
    host = sweep2array(sweep, "gain", info)
    np.testing.assert_array_equal(
        [to_int(scaler * float(g), soccfg.get_maxv(0), trunc=True) for g in host],
        expected,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scaler", [1.0, -1.0, 2.0])
def test_fractional_hardware_span_matches_full_scale_host_words(
    reverse, scaler
) -> None:
    soccfg = make_mock_soccfg()
    soccfg["gens"][0]["maxv_scale"] = 0.8
    start, stop = (1.0, -1.0) if reverse else (-1.0, 1.0)
    expected = [26212, 0, -26212] if reverse else [-26212, 0, 26212]
    sweep = SweepCfg(
        start=start / scaler,
        stop=stop / scaler,
        expts=3,
        step=(stop - start) / 2 / scaler,
    )
    info = {"soccfg": soccfg, "gen_ch": 0, "scaler": scaler}
    rounded = round_sweep_dict(sweep, "gain", info)
    np.testing.assert_array_equal(
        _compile_sweep_words(soccfg, rounded, scaler), expected
    )
    host = sweep2array(sweep, "gain", info)
    np.testing.assert_array_equal(
        [_compile_scalar_word(soccfg, scaler * float(g)) for g in host], expected
    )


@pytest.mark.parametrize(
    ("maxv", "maxv_scale", "middle_word"),
    [(32766, 1.0, 9829), (30000, 1.0, 9000), (32766, 0.9, 8846), (32766, 0.8, 7863)],
)
@pytest.mark.parametrize("scaler", [1.0, 2.0, 0.3, -1.0])
def test_custom_gain_array_preserves_shape_and_compiled_boundary_words(
    maxv, maxv_scale, middle_word, scaler
) -> None:
    soccfg = make_mock_soccfg()
    soccfg["gens"][0].update(maxv=maxv, maxv_scale=maxv_scale)
    limit_word = soccfg.get_maxv(0)
    requested = np.asarray([[-1.0, -0.3, -1.9 / limit_word], [0.0, 0.3, 1.0]]) / scaler
    expected = [[-limit_word, -middle_word, -1], [0, middle_word, limit_word]]
    actual = sweep2array(
        requested,
        "gain",
        {"soccfg": soccfg, "gen_ch": 0, "scaler": scaler},
        allow_array=True,
    )
    assert actual.shape == (2, 3)
    np.testing.assert_array_equal(
        actual, round_zcu_gain(requested, soccfg, 0, scaler=scaler)
    )
    words = [
        [_compile_scalar_word(soccfg, scaler * float(g)) for g in row] for row in actual
    ]
    np.testing.assert_array_equal(words, expected)
    assert actual[1, 0] == 0.0
    assert not np.signbit(actual[1, 0])


@pytest.mark.parametrize("sign", [1, -1])
def test_integer_boundary_keeps_original_input_truncation(sign) -> None:
    soccfg = make_mock_soccfg()
    soccfg["gens"][0]["maxv"] = 30000
    boundary = 59 / 30000
    requested = sign * np.asarray([boundary, np.nextafter(boundary, np.inf)])
    actual = round_zcu_gain(requested, soccfg, 0)
    words = [_compile_scalar_word(soccfg, float(g)) for g in actual]
    np.testing.assert_array_equal(words, [sign * 58, sign * 59])


@pytest.mark.parametrize(
    ("start_word", "stop_word"), [(-59, 0), (59, 0), (0, 59), (0, -59)]
)
@pytest.mark.parametrize("scaler", [1.0, -1.0, 0.3, 2.0])
def test_hardware_gain_span_near_zero_preserves_word_grid(
    start_word, stop_word, scaler
) -> None:
    soccfg = make_mock_soccfg()
    soccfg["gens"][0]["maxv"] = 30000
    start = (
        np.sign(start_word) * (abs(start_word) + 0.5) / (30000 * scaler)
        if start_word
        else 0.0
    )
    stop = (
        np.sign(stop_word) * (abs(stop_word) + 0.5) / (30000 * scaler)
        if stop_word
        else 0.0
    )
    sweep = SweepCfg(start=start, stop=stop, expts=2, step=stop - start)
    info = {"soccfg": soccfg, "gen_ch": 0, "scaler": scaler}
    rounded = round_sweep_dict(sweep, "gain", info)
    expected = [start_word, stop_word]
    np.testing.assert_array_equal(
        _compile_sweep_words(soccfg, rounded, scaler), expected
    )
    host = sweep2array(sweep, "gain", info)
    np.testing.assert_array_equal(
        [_compile_scalar_word(soccfg, scaler * float(g)) for g in host], expected
    )


def test_host_gain_axis_matches_compiled_words_with_frequency_sweep() -> None:
    soccfg = make_mock_soccfg(n_gens=3, n_readouts=1)
    gains = sweep2array(
        SweepCfg(start=0.1, stop=0.4, expts=4, step=0.1),
        "gain",
        {"soccfg": soccfg, "gen_ch": 1},
    )
    frequencies = SweepCfg(start=94.0, stop=108.0, expts=33, step=14.0 / 32)
    words = []
    for gain in gains:
        pulse = Pulse(
            "probe_pulse",
            PulseCfg(
                ch=1,
                nqz=1,
                freq=sweep2param("freq", frequencies),
                gain=float(gain),
                waveform=ConstWaveformCfg(length=0.1),
            ),
        )
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=0.1,
        ).build("readout")
        prog = compile_modules(
            soccfg, [pulse, readout], sweep=[("freq", frequencies)], reps=200
        )
        compiled_gain = prog.get_pulse_param(pulse.pulse_id, "gain", as_array=False)
        words.append(round(compiled_gain * soccfg.get_maxv(1)))

    np.testing.assert_array_equal(words, [3276, 6552, 9828, 13104])
    np.testing.assert_allclose(gains * soccfg.get_maxv(1), words, rtol=0, atol=2e-12)
