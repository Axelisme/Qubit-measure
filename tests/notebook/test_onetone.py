"""OneTone records use the common Notebook source and presentation boundary."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.onetone.freq import FreqAnalyzeOptions, FreqExp
from zcu_tools.experiment.v2.onetone.sa import SA_FreqExp, SA_FreqResult
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost

from tests.experiment.v2.onetone._support import (
    make_freq_cfg,
    make_freq_result,
    make_sa_cfg,
)


def test_load_b_then_analyze_save_a_retains_explicit_source_and_native_figure(
    tmp_path: Path,
) -> None:
    adapter = NotebookAdapter(host=NonPresentingHost())(FreqExp())
    original = RunRecord(cfg=make_freq_cfg(), result=make_freq_result(freq=6000.0))
    a = adapter.load(adapter.save(tmp_path / "a.hdf5", source=original))
    b = adapter.load(
        adapter.save(
            tmp_path / "b.hdf5",
            source=RunRecord(cfg=make_freq_cfg(), result=make_freq_result(freq=6020.0)),
        )
    )
    analysis = adapter.analyze(FreqAnalyzeOptions(edelay=0.021), source=a)
    assert analysis.source is a
    assert analysis.result.freq == pytest.approx(6000.0, abs=0.01)
    assert adapter.last_run is b
    assert tuple(analysis.figures) == ("fit",)
    handle = adapter.analysis_presentation
    assert handle is not None
    handle.release()
    figure = analysis.figures["fit"]
    figure.axes[2].set_title("User annotation")
    figure.savefig(tmp_path / "native.png")
    assert (tmp_path / "native.png").stat().st_size > 0
    saved = adapter.save(tmp_path / "saved-a.hdf5", source=a)
    reloaded = FreqExp().load(saved)
    assert reloaded.cfg == a.cfg
    np.testing.assert_array_equal(reloaded.result.signals, a.result.signals)
    previous = adapter.analysis
    with pytest.raises(ValueError, match="Invalid model type"):
        adapter.analyze(FreqAnalyzeOptions(model_type="invalid"), source=b)  # type: ignore[arg-type]
    assert adapter.analysis is previous
    assert adapter.last_run is b


def test_sa_none_options_is_a_successful_analysis_record(tmp_path: Path) -> None:
    adapter = NotebookAdapter(host=NonPresentingHost())(SA_FreqExp())
    source = RunRecord(
        cfg=make_sa_cfg(),
        result=SA_FreqResult(np.array([1.0, 2.0]), np.array([3 + 4j, -5j])),
    )
    record = adapter.analyze(None, source=source)
    assert record.source is source
    assert record.options is None and record.result is None
    np.testing.assert_array_equal(
        record.figures["fit"].axes[0].lines[0].get_ydata(), [5, 5]
    )
    handle = adapter.analysis_presentation
    assert handle is not None
    handle.release()
    record.figures["fit"].savefig(tmp_path / "sa-native.png")
    assert (tmp_path / "sa-native.png").stat().st_size > 0
