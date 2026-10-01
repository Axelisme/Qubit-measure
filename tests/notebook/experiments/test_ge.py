"""GE's explicit adopted-primary post tool and operation publication seam."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
)
from zcu_tools.notebook.experiments import GEPostAnalyzer, GEPrimaryRecord
from zcu_tools.plotting.plots import NonPresentingHost, Plots


def primary_record() -> GEPrimaryRecord:
    source = RunRecord[GE_Cfg, GE_Result](
        None,
        GE_Result(
            np.array([[-1 - 0.4j] * 10, [1 + 0.4j] * 10]),
            np.arange(10),
            np.array([0, 1]),
        ),
    )
    return AnalysisRecord(
        source=source,
        options=GEAnalyzeOptions(backend="center"),
        result=GEAnalysis(
            initial_state="ground",
            fidelity=1.0,
            theta=0.0,
            threshold=0.0,
            ge_s=0.2,
            g_center=-1 - 0.4j,
            e_center=1 + 0.4j,
            init_pops=np.eye(2),
        ),
        figures=Plots(NonPresentingHost()).finish(),
    )


def test_post_retains_explicit_primary_source_options_and_native_figure(
    tmp_path: Path,
) -> None:
    primary = primary_record()
    tool = GEPostAnalyzer(GE_Exp(), host=NonPresentingHost())
    options = GEPostAnalyzeOptions(radius=0.5)

    record = tool.analyze(primary, options)

    assert tool.analysis is record
    assert record.primary is primary
    assert record.source is primary.source
    assert record.options == options
    assert record.options is not options
    assert list(record.figures) == ["post"]
    np.testing.assert_allclose(record.result.confusion.matrix, np.eye(3))
    presentation = tool.analysis_plots
    assert presentation is not None
    assert presentation["post"] is record.figures["post"]
    presentation.release()
    record.figures["post"].savefig(tmp_path / "post.png")
    assert (tmp_path / "post.png").stat().st_size > 0
