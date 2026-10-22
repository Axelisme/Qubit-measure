"""Qt-free identity and pane replacement facts for analysis replies."""

from typing import Literal

from typing_extensions import TypedDict


class AnalysisPresentation(TypedDict):
    """Facts captured with one completed analysis result.

    run_id is the source Run/Load metadata identity, or None without metadata.
    stage is primary or post. figure_names lists the current pane's native names.
    figures_replaced means this publication replaced a nonempty figure collection
    in the same Run's same pane. The previous collection is no longer readable
    through that pane. This says nothing about saved files or other panes.
    Element positions and names remain in the experiment-owned summary.
    """

    run_id: str | None
    stage: Literal["primary", "post"]
    figure_names: list[str]
    figures_replaced: bool
