"""Qt-free presentation shared by the Analyze pane and remote replies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ..services.ports import TabSnapshot


class AnalysisPresentation(TypedDict):
    """Facts captured with one completed analysis result.

    run_id is the source Run/Load metadata identity, or None without metadata.
    stage is primary or post. figure_names lists the current pane's native names.
    figures_replaced means this publication replaced a nonempty figure collection
    in the same Run's same pane. The previous collection is no longer readable
    through that pane. This says nothing about saved files or other panes.
    summary_lines is the ordered operator text for Primary, omitted when empty
    or for Post. Element positions and names remain in the native summary.
    """

    run_id: str | None
    stage: Literal["primary", "post"]
    figure_names: list[str]
    figures_replaced: bool
    summary_lines: NotRequired[list[str]]


def project_analysis_presentation(
    snapshot: TabSnapshot, stage: Literal["primary", "post"]
) -> AnalysisPresentation:
    """Project the selected pane and its source Run from one tab snapshot.

    A missing pane has no figures, replacement, or summary lines. The result
    does not read live form inputs or regenerate the adapter's text.
    """
    pane = snapshot.analysis if stage == "primary" else snapshot.post_analysis
    presentation = AnalysisPresentation(
        run_id=None if snapshot.run is None else snapshot.run.acquisition_run_id,
        stage=stage,
        figure_names=list(() if pane is None else pane.figures or ()),
        figures_replaced=False if pane is None else pane.figures_replaced,
    )
    if (
        stage == "primary"
        and (primary := snapshot.analysis) is not None
        and primary.summary_lines
    ):
        presentation["summary_lines"] = list(primary.summary_lines)
    return presentation
