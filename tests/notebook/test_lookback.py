from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.lookback import (
    LookbackAnalyzeOptions,
    LookbackCfg,
    LookbackExp,
    LookbackResult,
)
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost

from tests.experiment.v2._lookback_support import make_lookback_cfg


def make_source(
    cfg: LookbackCfg | None, shift: float = 0.0
) -> RunRecord[LookbackCfg, LookbackResult]:
    return RunRecord(
        cfg=cfg,
        result=LookbackResult(
            np.array([0.0, 0.2, 0.4, 0.6, 0.8]) + shift,
            np.array([0j, 0.5j, 2j, 10j, 3j]),
        ),
    )


def test_old_source_retains_cfg_numbers_and_native_figure_after_load_b(
    tmp_path: Path,
) -> None:
    adapter = NotebookAdapter(LookbackExp(), host=NonPresentingHost())
    original = make_source(make_lookback_cfg(reps=7, trig_offset=0.25))
    path_a = adapter.save(original, tmp_path / "a.hdf5")
    source_a = adapter.load(path_a)
    initial = adapter.analyze(LookbackAnalyzeOptions(), source=source_a)
    presentation_a = adapter.analysis_presentation
    assert initial.result.predict_offset == pytest.approx(0.4)

    path_b = adapter.save(
        make_source(make_lookback_cfg(reps=3, trig_offset=1.5), shift=5.0),
        tmp_path / "b.hdf5",
    )
    source_b = adapter.load(path_b)
    assert adapter.last_run is source_b and adapter.analysis is None
    older = adapter.analyze(LookbackAnalyzeOptions(), source=source_a)
    assert adapter.last_run is source_b
    assert older.source is source_a
    assert older.result.predict_offset == pytest.approx(0.4)
    assert source_a.cfg is not None and source_a.cfg.reps == 7
    assert source_b.cfg is not None and source_b.cfg.reps == 3
    assert presentation_a is not None
    presentation_a.release()
    initial.figures["fit"].savefig(tmp_path / "old.png")
    assert (tmp_path / "old.png").stat().st_size > 0

    reloaded = adapter.load(adapter.save(source_a, tmp_path / "saved-a.hdf5"))
    assert reloaded.cfg == source_a.cfg
    np.testing.assert_array_equal(reloaded.result.signals, original.result.signals)
    assert reloaded.result.signals.dtype == np.complex128
    np.testing.assert_allclose(reloaded.result.times, original.result.times)


def test_analysis_failure_preserves_the_successful_record() -> None:
    adapter = NotebookAdapter(LookbackExp(), host=NonPresentingHost())
    successful = adapter.analyze(LookbackAnalyzeOptions(), source=make_source(None))
    invalid = RunRecord[LookbackCfg, LookbackResult](
        cfg=None, result=LookbackResult(np.array([]), np.array([], dtype=np.complex128))
    )
    with pytest.raises(ValueError, match="empty"):
        adapter.analyze(LookbackAnalyzeOptions(), source=invalid)
    assert adapter.analysis is successful
    assert successful.result.predict_offset == pytest.approx(0.4)
