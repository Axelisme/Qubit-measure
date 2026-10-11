import numpy as np
import pytest
from qick import QickConfig
from qick.asm_v2 import QickParam
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.utils.round_zcu import (
    apply_round,
    round_zcu_freq,
    round_zcu_gain,
    round_zcu_phase,
    round_zcu_time,
    sweep2array,
)
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    make_mock_soccfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


@pytest.fixture
def soccfg() -> QickConfig:
    return QickConfig(
        {
            "tprocs": [{"f_time": 430.08}],
            "gens": [{"f_fabric": 430.08}],
            "readouts": [{"f_output": 307.2}],
        }
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_time_sweep_matches_signed_qick_clock_grid(
    soccfg: QickConfig, reverse: bool
) -> None:
    start, stop = (3413.0 / 430.08, 13.0 / 430.08) if reverse else (0.03, 8.03)
    sweep = make_sweep(start, stop, expts=201)
    expected = (13 + 17 * np.arange(201)) / 430.08
    if reverse:
        expected = expected[::-1]

    actual = sweep2array(sweep, "time", {"soccfg": soccfg, "gen_ch": 0})

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual.min() > 0.0


def test_scalar_time_rounds_to_nearest_generator_cycle(soccfg: QickConfig) -> None:
    assert round_zcu_time(0.03, soccfg, gen_ch=0) == pytest.approx(13.0 / 430.08)


def test_single_point_time_sweep_uses_scalar_rounding(soccfg: QickConfig) -> None:
    sweep = make_sweep(0.03, 0.03, expts=1)
    np.testing.assert_allclose(
        sweep2array(sweep, "time", {"soccfg": soccfg}), [13.0 / 430.08]
    )


@pytest.mark.parametrize("stop", [0.031, 0.033])
def test_time_sweep_rejects_collapsed_hardware_steps(
    soccfg: QickConfig, stop: float
) -> None:
    sweep = make_sweep(0.03, stop, expts=10)
    with pytest.raises(ValueError, match="step"):
        sweep2array(sweep, "time", {"soccfg": soccfg})


def test_time_sweep_scales_before_hardware_rounding(soccfg: QickConfig) -> None:
    sweep = make_sweep(0.03, 0.23, expts=5)
    actual = sweep2array(sweep, "time", {"soccfg": soccfg, "gen_ch": 0, "scaler": 2.0})
    np.testing.assert_allclose(actual, (26 + 43 * np.arange(5)) / (2 * 430.08))


def test_time_sweep_uses_selected_readout_clock(soccfg: QickConfig) -> None:
    sweep = make_sweep(0.03, 0.23, expts=5)
    actual = sweep2array(sweep, "time", {"soccfg": soccfg, "ro_ch": 0})
    np.testing.assert_allclose(actual, (9 + 15 * np.arange(5)) / 307.2)


@pytest.mark.parametrize("scaler", [0.0, np.nan])
def test_time_sweep_rejects_invalid_scaler(soccfg: QickConfig, scaler: float) -> None:
    sweep = make_sweep(0.03, 0.23, expts=5)
    with pytest.raises(ValueError, match="scaler must be finite and nonzero"):
        sweep2array(sweep, "time", {"soccfg": soccfg, "scaler": scaler})


@pytest.mark.parametrize("mixer", [300.0, 317.5, 400.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_frequency_preview_matches_compiled_absolute_rf(mixer: float, reverse: bool):
    soccfg = make_mock_soccfg()
    soccfg["gens"][0].update(type="axis_sg_int4_v1", has_mixer=True)
    start, stop = 309.153205939, 309.153230019
    if reverse:
        start, stop = stop, start
    sweep = make_sweep(start, stop, expts=5)
    pulse = Pulse(
        "drive",
        PulseCfg(
            ch=0,
            nqz=1,
            mixer_freq=mixer,
            freq=QickParam(start, {"freq": stop - start}),
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.25),
        ),
    )
    prog = ModularProgramV2(soccfg, ProgramV2Cfg(), [pulse], sweep=[("freq", 5)])
    actual = sweep2array(
        sweep, "freq", {"soccfg": soccfg, "gen_ch": 0, "mixer_freq": mixer}
    )
    expected = prog.get_pulse_param(pulse.pulse_id, "freq", as_array=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert np.all(np.diff(actual) < 0) if reverse else np.all(np.diff(actual) > 0)


@pytest.mark.parametrize("freq", [-8.346794061, 5345.801980847])
@pytest.mark.parametrize("ro_ch", [None, 0])
def test_scalar_frequency_matches_compiler_signed_and_matched(freq, ro_ch):
    soccfg = make_mock_soccfg()
    pulse = Pulse(
        "drive",
        PulseCfg(
            ch=0,
            nqz=1,
            freq=freq,
            ro_ch=ro_ch,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.25),
        ),
    )
    prog = ModularProgramV2(soccfg, ProgramV2Cfg(), [pulse])
    expected = prog.get_pulse_param(pulse.pulse_id, "freq", as_array=False)
    assert round_zcu_freq(freq, soccfg, 0, ro_ch) == pytest.approx(
        expected, rel=0, abs=1e-10
    )


def test_single_frequency_point_and_scaled_negative_frequency():
    soccfg = make_mock_soccfg()
    requested = -8.346794061
    expected = round_zcu_freq(requested, soccfg, 0)
    assert round_zcu_freq(requested / 2, soccfg, 0, scaler=2.0) == pytest.approx(
        expected / 2, rel=0, abs=1e-12
    )
    np.testing.assert_allclose(
        sweep2array(
            make_sweep(requested, requested, expts=1),
            "freq",
            {"soccfg": soccfg, "gen_ch": 0},
        ),
        [expected],
        rtol=0,
        atol=1e-12,
    )


def test_frequency_sweep_rejects_collapsed_steps():
    with pytest.raises(ValueError, match="step"):
        sweep2array(
            make_sweep(309.0, 309.0 + 1e-7, expts=20),
            "freq",
            {"soccfg": make_mock_soccfg(), "gen_ch": 0},
        )


def test_vectorized_time_rounds_each_point_on_generator_clock(
    soccfg: QickConfig,
) -> None:
    actual = round_zcu_time(np.asarray([-0.03, 0.0, 0.03]), soccfg, gen_ch=0)

    np.testing.assert_allclose(actual, [-13.0 / 430.08, 0.0, 13.0 / 430.08])
    assert actual.shape == (3,)


def test_vectorized_frequency_matches_scalar_qick_conversion() -> None:
    soccfg = make_mock_soccfg()
    requested = np.asarray([[-8.346794061, 0.0], [100.123456789, 5345.801980847]])
    expected = np.asarray(
        [
            [round_zcu_freq(float(freq), soccfg, 0, ro_ch=0) for freq in row]
            for row in requested
        ]
    )

    actual = round_zcu_freq(requested, soccfg, 0, ro_ch=0)

    assert actual.shape == requested.shape
    assert actual[0, 0] < 0.0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_phase_rounding_preserves_scalar_and_vector_cycles() -> None:
    soccfg = make_mock_soccfg()
    requested = np.asarray([0.0, 90.0, 360.0, 720.0])
    resolution = soccfg.reg2deg(1, gen_ch=0) - soccfg.reg2deg(0, gen_ch=0)

    actual = round_zcu_phase(requested, soccfg, 0)

    assert actual.shape == requested.shape
    np.testing.assert_allclose(actual, requested, rtol=0, atol=resolution)
    for phase, rounded in zip(requested, actual, strict=True):
        assert round_zcu_phase(float(phase), soccfg, 0) == pytest.approx(
            float(phase), rel=0, abs=resolution
        )
        assert rounded == round_zcu_phase(float(phase), soccfg, 0)


def test_gain_rounding_truncates_signed_scalar_and_array_values() -> None:
    soccfg = make_mock_soccfg()
    maxv = soccfg.get_maxv(0)
    requested = np.asarray([-1.9 / maxv, 0.0, 1.9 / maxv])
    expected = np.asarray([-1.0 / maxv, 0.0, 1.0 / maxv])

    np.testing.assert_array_equal(round_zcu_gain(requested, soccfg, 0), expected)
    np.testing.assert_array_equal(
        round_zcu_gain(requested / 2.0, soccfg, 0, scaler=2.0), expected / 2.0
    )
    for gain, rounded in zip(requested, expected, strict=True):
        assert round_zcu_gain(float(gain), soccfg, 0) == rounded
        assert round_zcu_gain(float(gain / 2.0), soccfg, 0, scaler=2.0) == rounded / 2.0


@pytest.mark.parametrize(
    "sweep",
    [[0.03, -0.17, 0.24], np.asarray([0.03, -0.17, 0.24])],
    ids=["list", "array"],
)
def test_custom_sweep_array_policy_preserves_values(
    sweep: list[float] | np.ndarray,
) -> None:
    actual = sweep2array(sweep, allow_array=True)

    np.testing.assert_array_equal(actual, [0.03, -0.17, 0.24])
    assert actual.dtype == np.float64
    with pytest.raises(ValueError, match="Custom sweep is not allowed:"):
        sweep2array(sweep, allow_array=False)


def test_custom_time_sweep_quantizes_each_array_point(soccfg: QickConfig) -> None:
    requested = np.asarray([-0.03, 0.0, 0.03])

    actual = sweep2array(
        requested, "time", {"soccfg": soccfg, "gen_ch": 0}, allow_array=True
    )

    np.testing.assert_allclose(actual, [-13.0 / 430.08, 0.0, 13.0 / 430.08])
    assert actual.shape == requested.shape


def test_apply_round_dispatches_identity_and_time(soccfg: QickConfig) -> None:
    requested = np.asarray([-0.03, 0.0, 0.03])

    assert apply_round(requested, "none", {}) is requested
    assert apply_round(0.03, "none", {}) == 0.03
    assert apply_round(0.03, "time", {"soccfg": soccfg, "gen_ch": 0}) == pytest.approx(
        13.0 / 430.08
    )
    np.testing.assert_allclose(
        apply_round(requested, "time", {"soccfg": soccfg, "gen_ch": 0}),
        [-13.0 / 430.08, 0.0, 13.0 / 430.08],
    )
