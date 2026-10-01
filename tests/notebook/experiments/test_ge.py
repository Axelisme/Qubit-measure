"""GE's explicit adopted-primary post tool and operation publication seam."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.records import AnalysisRecord, RunRecord
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalysis,
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


class PostOnlyCore(GE_Exp):
    def __init__(self) -> None:
        self.fail = False

    def analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        options: GEAnalyzeOptions,
        *,
        plots: Plots,
    ) -> GEAnalysis:
        raise AssertionError("Post must not refit the adopted primary")

    def post_analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        primary: GEAnalysis,
        options: GEPostAnalyzeOptions,
        *,
        plots: Plots,
    ) -> GEPostAnalysis:
        result = super().post_analyze(source, primary, options, plots=plots)
        object.__setattr__(options, "radius", 99.0)
        if self.fail:
            raise ValueError("Post producer failed")
        return result


class FailingHost(NonPresentingHost):
    def __init__(self) -> None:
        self.fail_present = False
        self.fail_release = False
        self.released: list[Figure] = []

    def present(self, figure: Figure) -> None:
        if self.fail_present:
            raise ValueError("Post finish failed")

    def release(self, figure: Figure) -> None:
        self.released.append(figure)
        if self.fail_release:
            raise OSError("Post cleanup failed")


def test_post_retains_explicit_primary_source_options_and_native_figure(
    tmp_path: Path,
) -> None:
    primary = primary_record()
    tool = GEPostAnalyzer(PostOnlyCore(), host=NonPresentingHost())
    options = GEPostAnalyzeOptions(radius=0.5)

    record = tool.analyze(primary, options)

    assert tool.analysis is record
    assert record.primary is primary
    assert record.source is primary.source
    assert record.options == options
    assert record.options is not options
    assert record.result.confusion.radius == 0.5
    object.__setattr__(options, "radius", 3.0)
    assert record.options.radius == 0.5
    assert list(record.figures) == ["post"]
    np.testing.assert_allclose(record.result.confusion.matrix, np.eye(3))
    presentation = tool.analysis_plots
    assert presentation is not None
    assert presentation["post"] is record.figures["post"]
    presentation.release()
    record.figures["post"].savefig(tmp_path / "post.png")
    assert (tmp_path / "post.png").stat().st_size > 0


@pytest.mark.parametrize("failure", ["producer", "finish"])
def test_failed_post_preserves_success_and_releases_only_its_diagnostic(
    failure: str, tmp_path: Path
) -> None:
    core, host = PostOnlyCore(), FailingHost()
    tool = GEPostAnalyzer(core, host=host)
    primary = primary_record()
    previous = tool.analyze(primary, GEPostAnalyzeOptions(radius=0.5))
    previous_plots = tool.analysis_plots
    core.fail = failure == "producer"
    host.fail_present = failure == "finish"

    with pytest.raises(ValueError, match="Post .* failed"):
        tool.analyze(primary, GEPostAnalyzeOptions(radius=0.7))

    assert tool.analysis is previous
    assert tool.analysis_plots is previous_plots
    assert len(host.released) == 1
    assert host.released[0] is not previous.figures["post"]
    previous.figures["post"].savefig(tmp_path / "old-post.png")
    assert (tmp_path / "old-post.png").stat().st_size > 0


def test_post_producer_and_cleanup_errors_both_remain_observable() -> None:
    core, host = PostOnlyCore(), FailingHost()
    tool = GEPostAnalyzer(core, host=host)
    primary = primary_record()
    previous = tool.analyze(primary, GEPostAnalyzeOptions(radius=0.5))
    previous_plots = tool.analysis_plots
    core.fail = host.fail_release = True

    with pytest.raises(BaseExceptionGroup) as raised:
        tool.analyze(primary, GEPostAnalyzeOptions(radius=0.7))

    producer, cleanup = raised.value.exceptions
    assert isinstance(producer, ValueError)
    assert str(producer) == "Post producer failed"
    assert isinstance(cleanup, BaseExceptionGroup)
    assert len(cleanup.exceptions) == 1
    assert isinstance(cleanup.exceptions[0], OSError)
    assert str(cleanup.exceptions[0]) == "Post cleanup failed"
    assert len(host.released) == 1
    assert tool.analysis is previous
    assert tool.analysis_plots is previous_plots
