"""Independent Notebook flux picks retain explicit sources and native figures."""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.line_state import FluxPickAnalysis, FluxPickState
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.onetone.flux_dep import FluxDepCfg, FluxDepResult
from zcu_tools.notebook.experiments import FluxDepAnalyzer, FluxDepPickerOptions
from zcu_tools.plotting.plots import NonPresentingHost


def make_source() -> RunRecord[FluxDepCfg, FluxDepResult]:
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    signals = np.asarray(
        np.sin(values[:, None] * 7 + freqs[None, :] * 9)
        + 1j * np.cos(values[:, None] * 3 - freqs[None, :] * 7),
        dtype=np.complex128,
    )
    return RunRecord(cfg=None, result=FluxDepResult(values, freqs, signals))


def test_done_publishes_source_options_numeric_result_and_named_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "zcu_tools.notebook.experiments.flux_dep.ipython_display.display",
        lambda _widget: None,
    )
    analyzer = FluxDepAnalyzer(NonPresentingHost())
    source = make_source()
    original_signals = source.result.signals.copy()
    control = analyzer.start(
        source, FluxDepPickerOptions(-0.2, 0.3, magnitude_only=True)
    )
    try:
        assert isinstance(control.widget, widgets.VBox)
        assert control.positions() == pytest.approx((-0.2, 0.3))
        assert control.record is None
        assert analyzer.analysis is None
        control.set_positions(-0.1, 0.4)
        control.conjugate_checkbox.value = True

        record = control.done()

        assert control.is_finished
        assert analyzer.analysis is record
        assert record.source is source
        assert record.options == FluxPickState(
            flux_half=-0.1, flux_int=0.4, conjugate=True, magnitude_only=True
        )
        assert record.result == FluxPickAnalysis(-0.1, 0.4, 1.0)
        assert control.record is record
        assert control.plots is analyzer.analysis_plots
        assert list(record.figures) == ["pick"]
        assert record.figures["pick"] is not control.figure
        record.figures["pick"].savefig(tmp_path / "onetone-pick.png")
        assert (tmp_path / "onetone-pick.png").stat().st_size > 0
        np.testing.assert_array_equal(source.result.signals, original_signals)
        with pytest.raises(RuntimeError, match="finished"):
            control.done()
    finally:
        if not control.is_finished:
            control.cancel()
        if analyzer.analysis_plots is not None:
            analyzer.analysis_plots.release()
