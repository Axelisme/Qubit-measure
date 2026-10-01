"""OneTone notebook publishes only completed, numeric interactive picks."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import pytest
from ipympl.backend_nbagg import Canvas, Toolbar
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepAnalysis,
    FluxDepAnalyzeOptions,
    FluxDepCfg,
    FluxDepModuleCfg,
    FluxDepResult,
    FluxDepSweepCfg,
)
from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepExp as FluxDepCore,
)
from zcu_tools.experiment.v2.runtime.schedule import (
    ScheduleOutcomeError,
    ScheduleStep,
    SignalBuffer,
)
from zcu_tools.notebook.experiments import FluxDepNotebookExp
from zcu_tools.notebook.plotting import NotebookPlotHost
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sweep import SweepCfg


def make_source() -> FluxDepResult:
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    signals = np.asarray(
        np.sin(values[:, None] * 7 + freqs[None, :] * 9)
        + 1j * np.cos(values[:, None] * 3 - freqs[None, :] * 7),
        dtype=np.complex128,
    )
    return FluxDepResult(values, freqs, signals)


def make_cfg() -> FluxDepCfg:
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.2,
        freq=7000.0,
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    )
    return FluxDepCfg(
        reps=1,
        rounds=1,
        dev={},
        modules=FluxDepModuleCfg(
            readout=PulseReadoutCfg(
                pulse_cfg=pulse,
                ro_cfg=DirectReadoutCfg(
                    ro_ch=0, gen_ch=0, ro_length=1.0, ro_freq=7000.0, trig_offset=0.0
                ),
            ),
        ),
        sweep=FluxDepSweepCfg(
            flux=SweepCfg(start=-0.5, stop=0.5, step=0.125, expts=9),
            freq=SweepCfg(start=4.8, stop=5.4, step=0.1, expts=7),
        ),
    )


def test_canonical_save_load_resets_current_analysis_but_retains_old_pick(
    tmp_path: Path,
) -> None:
    measured = make_source()
    cfg = make_cfg()
    source = FluxDepResult(measured.values, measured.freqs, measured.signals, cfg)
    original = tmp_path / "original.hdf5"
    FluxDepCore().save(source, original)
    exp = FluxDepNotebookExp(present=False)
    loaded = exp.load(original)
    np.testing.assert_array_equal(loaded.signals, source.signals)
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.model_dump() == cfg.model_dump()
    exp.analyze(flux_half=-0.2, flux_int=0.3).done()
    previous = exp.analysis
    old_pick = exp.analysis_plots
    with pytest.raises(FileNotFoundError):
        exp.load(tmp_path / "missing.hdf5")
    assert exp.analysis is previous
    assert exp.last_result is loaded
    saved = tmp_path / "saved.hdf5"
    exp.save(saved, comment="Notebook flux map")
    replaced = exp.load(saved)
    assert exp.last_result is replaced
    assert exp.analysis is None
    assert previous is not None and previous.source is loaded
    np.testing.assert_array_equal(replaced.values, source.values)
    np.testing.assert_array_equal(replaced.freqs, source.freqs)
    old_pick["pick"].savefig(tmp_path / "old-pick.png")
    assert (tmp_path / "old-pick.png").stat().st_size > 0
    assert old_pick is previous.plots


def test_failed_run_keeps_published_analysis_and_releases_its_partial_plot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from zcu_tools.experiment.context import QickContext

    exp = FluxDepNotebookExp(present=False)
    source = make_source()
    exp.analyze(source, flux_half=-0.2, flux_int=0.3).done()
    old = exp.analysis
    old_pick = exp.analysis_plots
    captures = []

    def fail_run(self: FluxDepCore, cfg: FluxDepCfg, *, context: QickContext):
        del self, cfg
        viewer = context.plots.liveplot_2d_with_line(
            "measurement", "Flux", "Frequency", uniform=False
        )
        viewer.update(
            source.values,
            source.freqs,
            np.asarray(np.abs(source.signals), dtype=np.float64),
        )
        captures.append(context.plots)
        raise RuntimeError("simulated acquisition stopped")

    monkeypatch.setattr(FluxDepCore, "run", fail_run)
    with pytest.raises(RuntimeError, match="simulated acquisition stopped"):
        exp.run("sim-soc", "sim-cfg", make_cfg())
    assert exp.last_result is None
    assert exp.analysis is old
    assert exp.analysis_plots is old_pick
    assert len(captures) == 1
    captures[0]["measurement"].savefig(tmp_path / "stopped-map.png")
    assert (tmp_path / "stopped-map.png").stat().st_size > 0
