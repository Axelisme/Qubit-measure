import numpy as np
import pytest
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.time_sweep import plan_time_sweep
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    PulseCfg,
    make_mock_soccfg,
    sweep2param,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


@pytest.mark.parametrize("reverse", [False, True])
def test_planned_time_window_preserves_span_and_compiled_coordinates(
    reverse: bool,
) -> None:
    soccfg = make_mock_soccfg()
    cycle = soccfg.cycles2us(1, gen_ch=0)
    start, stop = (193, 20) if reverse else (20, 193)
    requested = make_sweep(start * cycle, stop * cycle, expts=101)

    planned = plan_time_sweep(requested, soccfg, gen_ch=0)

    assert planned.expts == 87
    assert abs(planned.step) == pytest.approx(2 * cycle)
    assert planned.stop == pytest.approx((21 if reverse else 192) * cycle)
    assert abs(requested.stop - planned.stop) < abs(planned.step)
    pulse = Pulse(
        "drive",
        PulseCfg(
            ch=0,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=sweep2param("length", planned)),
        ),
    )
    prog = ModularProgramV2(soccfg, [pulse], sweep=[("length", planned)])
    compiled = prog.get_pulse_param(pulse.pulse_id, "total_length", as_array=True)
    expected = (
        (193 - 2 * np.arange(87)) if reverse else (20 + 2 * np.arange(87))
    ) * cycle
    np.testing.assert_allclose(compiled, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        sweep2array(planned, "time", {"soccfg": soccfg, "gen_ch": 0}),
        compiled,
        rtol=0,
        atol=1e-12,
    )
    assert requested.expts == 101


@pytest.mark.parametrize("cycles,points", [(0.5, 101), (1.2, 101), (0.0, 1)])
def test_unusable_time_plan_reports_clock_and_available_points(cycles, points) -> None:
    soccfg = make_mock_soccfg()
    cycle = soccfg.cycles2us(1, gen_ch=0)
    # At 20.51 cycles the rounded start leaves < 1 cycle even for a 1.2-cycle span.
    requested = make_sweep(20.51 * cycle, (20.51 + cycles) * cycle, expts=points)

    with pytest.raises(ValueError, match=r"span=.*cycle=.*max_available_points=1\b"):
        plan_time_sweep(requested, soccfg, gen_ch=0)


@pytest.mark.parametrize("scaler", [0.0, np.nan, np.inf])
def test_time_plan_rejects_invalid_scaler(scaler: float) -> None:
    with pytest.raises(ValueError, match="scaler"):
        plan_time_sweep(
            make_sweep(0.05, 0.56, expts=101), make_mock_soccfg(), scaler=scaler
        )


def test_scaled_readout_clock_plan_matches_preview() -> None:
    soccfg = make_mock_soccfg()
    cycle = soccfg.cycles2us(1, ro_ch=0) / 2
    requested = make_sweep(20 * cycle, 193 * cycle, expts=101)
    planned = plan_time_sweep(requested, soccfg, ro_ch=0, scaler=2)
    actual = sweep2array(planned, "time", {"soccfg": soccfg, "ro_ch": 0, "scaler": 2})
    np.testing.assert_allclose(
        actual, (20 + 2 * np.arange(87)) * cycle, rtol=0, atol=1e-12
    )
