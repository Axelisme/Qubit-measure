"""OneTone notebook publishes only completed, numeric interactive picks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ipywidgets as widgets
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
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
from zcu_tools.notebook.experiments import FluxDepNotebookExp
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


def test_done_publishes_source_options_numeric_result_and_named_figure(
    tmp_path,
) -> None:
    exp = FluxDepNotebookExp(present=False)
    source = make_source()
    control = exp.analyze(source, flux_half=-0.2, flux_int=0.3, magnitude_only=True)
    assert isinstance(control.widget, widgets.VBox)
    assert control.positions() == pytest.approx((-0.2, 0.3))
    assert control.result is None
    assert exp.analysis is None

    control.done_button.click()
    assert control.is_finished
    record = exp.analysis
    assert record is not None
    assert record.source is source
    assert record.options == FluxDepAnalyzeOptions(-0.2, 0.3, magnitude_only=True)
    assert record.result == FluxDepAnalysis(-0.2, 0.3, 1.0)
    assert control.result is record.result
    assert control.plots is exp.analysis_plots
    assert list(record.plots) == ["pick"]
    record.plots["pick"].savefig(tmp_path / "onetone-pick.png")
    assert (tmp_path / "onetone-pick.png").stat().st_size > 0
    with pytest.raises(RuntimeError, match="finished"):
        control.done()


def test_invalid_done_and_cancel_leave_old_analysis_and_editable_control() -> None:
    exp = FluxDepNotebookExp(present=False)
    first = exp.analyze(make_source(), flux_half=-0.2, flux_int=0.3)
    first.done()
    old = exp.analysis

    next_control = exp.analyze(make_source(), flux_half=0.0, flux_int=0.0)
    with pytest.raises(ValueError, match="separated"):
        next_control.done()
    assert not next_control.is_finished
    assert exp.analysis is old
    next_control.set_positions(-0.1, 0.4)
    next_control.cancel_button.click()
    assert next_control.is_finished
    assert exp.analysis is old
    with pytest.raises(RuntimeError, match="finished"):
        next_control.done()


@pytest.mark.parametrize("reverse_flux", [False, True])
def test_simulated_run_publishes_typed_measurement_without_hardware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reverse_flux: bool
) -> None:
    import zcu_tools.experiment.v2.onetone.flux_dep as core_module

    cfg = make_cfg()
    source = make_source()
    if reverse_flux:
        cfg.sweep.flux = SweepCfg(start=0.5, stop=-0.5, step=-0.125, expts=9)
        source = FluxDepResult(
            source.values[::-1].copy(), source.freqs, source.signals[::-1].copy()
        )
    before = cfg.model_dump()

    class FakeBuilder:
        def __init__(self, buffer: Any, row: int) -> None:
            self.buffer = buffer
            self.row = row

        def add_reset(self, *_args: Any) -> FakeBuilder:
            return self

        def add(self, *_args: Any) -> FakeBuilder:
            return self

        def declare_sweep(self, *_args: Any) -> FakeBuilder:
            return self

        def build_and_acquire(self, **_kwargs: Any) -> None:
            self.buffer.at(self.row).set(source.signals[self.row])

    class FakeSchedule:
        def __init__(self, config: FluxDepCfg, buffer: Any) -> None:
            self.config = config
            self.buffer = buffer

        def __enter__(self) -> FakeSchedule:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def scan(self, _field: str, values: list[float]):
            for row, value in enumerate(values):
                yield (
                    row,
                    SimpleNamespace(
                        cfg=self.config,
                        value=value,
                        prog_builder=lambda _soc, _cfg, row=row: FakeBuilder(
                            self.buffer, row
                        ),
                    ),
                )

    monkeypatch.setattr(core_module, "Schedule", FakeSchedule)
    monkeypatch.setattr(core_module, "set_flux_in_dev_cfg", lambda *_args: None)
    monkeypatch.setattr(core_module, "setup_devices", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        core_module,
        "sweep2array",
        lambda _sweep, field=None, *_args, **_kwargs: (
            source.freqs if field == "freq" else source.values
        ),
    )
    exp = FluxDepNotebookExp(present=False)
    result = exp.run("simulated-soc", "simulated-soccfg", cfg)
    assert result is exp.last_result
    assert result.cfg_snapshot is not None
    assert result.cfg_snapshot.model_dump() == before
    assert cfg.model_dump() == before
    np.testing.assert_array_equal(result.signals, source.signals)
    np.testing.assert_array_equal(result.values, source.values)
    np.testing.assert_array_equal(result.freqs, source.freqs)
    plots = exp.run_plots
    assert plots is not None and tuple(plots) == ("measurement",)
    assert len(plots["measurement"].axes) == 2
    plots.release()
    plots["measurement"].savefig(tmp_path / "flux-map.png")
    assert (tmp_path / "flux-map.png").stat().st_size > 0


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


@pytest.mark.parametrize("fail_on", [1, 2])
def test_failed_notebook_publication_cleans_up_widgets_and_keeps_record(
    monkeypatch: pytest.MonkeyPatch, fail_on: int
) -> None:
    exp = FluxDepNotebookExp()
    source = make_source()
    existing_widgets = set(widgets.Widget.widgets)
    calls = 0

    def fail_display(_widget: object) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise RuntimeError("frontend publisher failed")

    monkeypatch.setattr("IPython.display.display", fail_display)
    with pytest.raises(RuntimeError, match="frontend publisher failed"):
        exp.analyze(source, flux_half=-0.2, flux_int=0.3)
    assert exp.analysis is None
    assert set(widgets.Widget.widgets) == existing_widgets


def test_default_notebook_host_drags_and_releases_preview_widgets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    displayed: list[object] = []
    monkeypatch.setattr("IPython.display.display", displayed.append)
    existing_widgets = set(widgets.Widget.widgets)
    exp = FluxDepNotebookExp()
    control = exp.analyze(make_source(), flux_half=-0.2, flux_int=0.3)
    try:
        assert displayed == [control.widget, control.figure.canvas]
        canvas = control.figure.canvas
        control.half_button.click()
        canvas.draw()
        ax = control.figure.axes[0]
        pixel_x, pixel_y = ax.transData.transform((-0.1, 5.1))
        canvas.callbacks.process(
            "motion_notify_event",
            MouseEvent("motion_notify_event", canvas, pixel_x, pixel_y),
        )
        assert control.positions()[0] == pytest.approx(-0.1, abs=0.01)
        preview = control.figure
        control.done_button.click()
        assert control.is_finished
        assert exp.analysis is not None
        assert exp.analysis.result.flux_half == pytest.approx(-0.1, abs=0.01)
        assert preview.canvas is not canvas
        exp.analysis_plots["pick"].savefig(tmp_path / "interactive-pick.png")
        assert (tmp_path / "interactive-pick.png").stat().st_size > 0
    finally:
        if not control.is_finished:
            control.cancel()
        if exp.analysis is not None:
            exp.analysis_plots.release()
    assert set(widgets.Widget.widgets) == existing_widgets
