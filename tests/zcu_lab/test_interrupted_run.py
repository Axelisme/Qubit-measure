"""Registered/core run boundaries retain owner buffers after an interruption."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qick import QickConfig
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.measure.adapter import RunRequest
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soccfg

from zcu_lab.definitions import register_all
from zcu_lab.v2.lookback.core import LookbackCfg, LookbackExp


class InterruptBeforeAcquire(StopSignal):
    def is_set(self) -> bool:
        raise KeyboardInterrupt("interrupted before the first acquisition")


class InterruptAfterFrame(NonPresentingHost):
    def __init__(self) -> None:
        self.frame: NDArray[np.float64] | None = None

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        if not final and self.frame is None:
            self.frame = np.array(figure.axes[0].lines[0].get_ydata())
            raise KeyboardInterrupt("interrupted after the first buffered frame")


class InterruptAxisConfig(QickConfig):
    def cycles2us(self, cycles, gen_ch=None, ro_ch=None):
        if isinstance(cycles, np.ndarray):
            raise KeyboardInterrupt("interrupted while configuring the time axis")
        return super().cycles2us(cycles, gen_ch=gen_ch, ro_ch=ro_ch)


def readout_cfg() -> dict[str, ExperimentCfgValue]:
    return {
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 0.1},
            "ch": 0,
            "nqz": 1,
            "freq": 100.0,
            "gain": 0.2,
        },
        "ro_cfg": {
            "ro_ch": 0,
            "ro_freq": 100.0,
            "ro_length": 0.1,
            "trig_offset": 0.25,
        },
    }


@pytest.mark.parametrize(
    "before_acquire", [True, False], ids=["before-acquire", "after-frame"]
)
def test_registered_run_retains_interrupted_buffer(before_acquire: bool) -> None:
    registry = Registry()
    register_all(registry)
    adapter = registry.create("fake/freq")
    host = InterruptAfterFrame()
    plots = Plots(NonPresentingHost() if before_acquire else host)
    stop = InterruptBeforeAcquire() if before_acquire else StopSignal()
    context = RunContext(
        soc=None, soccfg=None, plots=plots, devices={}, cancel_signal=stop
    )
    raw_cfg: dict[str, ExperimentCfgValue] = {
        "modules": {"readout": readout_cfg()},
        "sweep": {"freq": {"start": 4000.0, "stop": 5000.0, "expts": 3, "step": 500.0}},
        "reps": 1,
        "rounds": 2,
    }
    try:
        record = adapter.run(
            RunRequest(soc=None, soccfg=None, cfg_context=None),
            raw_cfg,
            context=context,
        )
    except UnboundLocalError:
        pytest.fail("an interrupted registered run must return its owner buffer")
    finally:
        plots.finish(present=False)
        plots.release()

    assert isinstance(record, RunRecord), (
        "an interrupted run must return a record, not None"
    )
    np.testing.assert_array_equal(record.result.freqs, [4000.0, 4500.0, 5000.0])
    assert record.result.signals.shape == (3,)
    if before_acquire:
        assert np.isnan(record.result.signals).all()
    else:
        assert np.isfinite(record.result.signals).all()
        np.testing.assert_array_equal(np.abs(record.result.signals), host.frame)
    assert stop.error is not None
    assert stop.error.status == "interrupted"


def test_lookback_axis_configuration_interrupt_propagates() -> None:
    cfg = LookbackCfg.model_validate({"modules": {"readout": readout_cfg()}, "reps": 1})
    soccfg = InterruptAxisConfig(make_mock_soccfg().get_cfg())
    plots = Plots(NonPresentingHost())
    context = RunContext(
        soc=None, soccfg=soccfg, plots=plots, devices={}, cancel_signal=StopSignal()
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="configuring the time axis"):
            LookbackExp().run(cfg, context=context)
    finally:
        plots.finish(present=False)
        plots.release()


def test_lookback_interrupted_acquire_retains_configured_axis_and_buffer() -> None:
    cfg = LookbackCfg.model_validate({"modules": {"readout": readout_cfg()}, "reps": 1})
    plots = Plots(NonPresentingHost())
    stop = InterruptBeforeAcquire()
    context = RunContext(
        soc=None, soccfg=make_mock_soccfg(), plots=plots, devices={}, cancel_signal=stop
    )
    try:
        result = LookbackExp().run(cfg, context=context)
    finally:
        plots.finish(present=False)
        plots.release()
    assert result.times[0] == 0.25
    assert result.times[-1] > result.times[0]
    assert result.signals.shape == result.times.shape
    assert np.isnan(result.signals).all()
    assert stop.error is not None
    assert stop.error.status == "interrupted"
