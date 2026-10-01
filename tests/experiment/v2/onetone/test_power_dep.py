"""Gain/frequency ordering, typed SNR stopping and canonical maps."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from zcu_tools.datafile import load_labber_data
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.onetone.power_dep import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_tools.experiment.v2.runtime.schedule import ScheduleStep
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.mocksoc import make_mock_soc

from tests.experiment.v2.onetone._support import make_power_cfg


def test_mock_soc_run_preserves_cfg_and_gain_outer_frequency_inner() -> None:
    cfg = make_power_cfg()
    before = cfg.model_copy(deep=True)
    soc, soccfg = make_mock_soc()
    plots = Plots(NonPresentingHost())
    result = PowerDepExp().run(cfg, context=QickContext(soc, soccfg, plots))
    figures = plots.finish()
    assert cfg == before
    assert result.signals.shape == (len(result.gains), len(result.freqs)) == (3, 9)
    assert result.signals.dtype == np.complex128
    assert np.all(np.isfinite(result.signals))
    assert np.all(np.diff(result.freqs) > 0)
    np.testing.assert_allclose(result.gains, [0.1, 0.2, 0.3], atol=1e-4)
    assert tuple(figures) == ("measurement",)
    assert len(figures["measurement"].axes) == 2
    plots.release()


@pytest.mark.parametrize("threshold", [None, 0.01])
@pytest.mark.parametrize("abort_after", [None, 2])
def test_snr_callback_observes_current_row_and_abort_preserves_nan_rows(
    monkeypatch: pytest.MonkeyPatch, threshold: float | None, abort_after: int | None
) -> None:
    cfg = make_power_cfg(earlystop_snr=threshold)
    row = np.exp(-(np.linspace(-2.0, 2.0, 9) ** 2)).astype(np.complex128) * (1 + 1j)
    controls: list[bool] = []
    gains: list[float] = []

    class Acquisition:
        def __init__(self, step: ScheduleStep[PowerDepCfg, float, object]) -> None:
            self.step = step

        def add_reset(self, *_args: Any) -> Acquisition:
            return self

        def add(self, *_args: Any) -> Acquisition:
            return self

        def declare_sweep(self, *_args: Any) -> Acquisition:
            return self

        def build_and_acquire(self, *, stop_condition: Callable[[], bool]) -> None:
            gains.append(self.step.value)
            self.step.set_data(row)
            controls.append(stop_condition())
            if abort_after is not None and len(controls) == abort_after:
                self.step.set_stop()

    def builder(
        step: ScheduleStep[PowerDepCfg, float, object], _soc: object, _soccfg: object
    ) -> Acquisition:
        return Acquisition(step)

    monkeypatch.setattr(ScheduleStep, "prog_builder", builder)
    soc, soccfg = make_mock_soc()
    plots = Plots(NonPresentingHost())
    result = PowerDepExp().run(cfg, context=QickContext(soc, soccfg, plots))
    plots.finish()
    visited = 3 if abort_after is None else abort_after
    np.testing.assert_allclose(gains, result.gains[:visited])
    assert controls == [threshold is not None] * visited
    np.testing.assert_allclose(result.signals[:visited], np.tile(row, (visited, 1)))
    if abort_after is not None:
        assert np.all(np.isnan(result.signals[abort_after:]))
    assert cfg.earlystop_snr == threshold
    plots.release()


def test_canonical_roundtrip_preserves_complex_map_axes_and_snr_control(
    tmp_path: Path,
) -> None:
    result = PowerDepResult(
        np.array([0.1, 0.2]),
        np.array([5970.0, 6000.0, 6030.0]),
        np.array([[1 + 2j, 3j, 4 + 5j], [6j, 7j, np.nan + 1j * np.nan]]),
    )
    source = RunRecord(cfg=make_power_cfg(earlystop_snr=2.5), result=result)
    core = PowerDepExp()
    path = tmp_path / "map.hdf5"
    core.save(source, path)
    loaded = core.load(path)
    assert loaded.cfg == source.cfg
    np.testing.assert_allclose(loaded.result.signals, result.signals, equal_nan=True)
    np.testing.assert_array_equal(loaded.result.gains, result.gains)
    np.testing.assert_allclose(loaded.result.freqs, result.freqs)
    disk = load_labber_data(str(path))
    np.testing.assert_allclose(disk.axes[0].values, result.freqs * 1e6)
    np.testing.assert_array_equal(disk.axes[1].values, result.gains)
