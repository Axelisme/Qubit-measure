"""Adapter-owned content handover; all paths, formats and writes stay framework-owned."""

from typing import Literal, Protocol

from zcu_tools.experiment.artifacts import AnalysisItem, DataItem
from zcu_tools.resources.entry import ArtifactKey


class ArtifactHandover(Protocol):
    """Content producers called synchronously on the GUI owner thread during Save.

    Results are opaque to the framework. Producers may validate their own types,
    but must not read live context, select paths/formats, write files, encode
    figures or mutate result arrays. Figures come from committed pane plots.
    """

    def data_item(self, key: ArtifactKey, result: object, *, comment: str) -> DataItem:
        """Hand over run content for the supplied data identity and raw annotation.

        key is framework-assigned; result is the exact committed run handle.
        Use DataItem.from_run for a typed record/spec, DataItem for already
        prepared content, or DataItem.failed for an explicit preparation error.
        A raised preparation exception also becomes failed data outputs without
        blocking independent figures or analysis. Missing cfg must not be filled
        from current Form input. No I/O or acquisition metadata capture occurs.
        """
        ...

    def analysis_item(
        self,
        key: ArtifactKey,
        result: object,
        parameters: object | None,
        *,
        pane: Literal["analysis", "post_analysis"],
    ) -> AnalysisItem:
        """Hand over the supplied pane's committed summary and historical inputs.

        key is framework-assigned, result is its committed output and parameters
        is the input instance used for that output, never an edited draft.
        Return AnalysisItem with an AnalysisDocument containing finite JSON, not
        bytes or figures. Invalid content raises TypeError/ValueError before any
        annotation or file mutation. No live-context lookup or I/O is permitted.
        """
        ...
