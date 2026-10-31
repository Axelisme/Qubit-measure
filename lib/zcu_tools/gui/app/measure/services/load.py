from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.datafile import load
from zcu_tools.experiment.metadata import RunMetadataContent
from zcu_tools.experiment.native_persistence import metadata_content_from_container
from zcu_tools.gui.app.measure.services.pane_commit import commit_run
from zcu_tools.gui.app.measure.state import RunPaneState
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.parameter_capture import restore_parameters
from zcu_tools.gui.session.parameters import ParameterContext

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.artifacts import LayoutPlanner
    from zcu_tools.gui.app.measure.state import State


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadTabResultOutcome:
    """Committed Load facts, independent of the tab's editable Form input.

    tab_id names the receiving tab; data_path is the selected source file.
    result_type is the opaque adapter record's class name. cfg_issue is None when
    typed cfg is available, or the adapter's readable unavailability reason.
    has_analyze_params reports successful
    analysis preparation; analysis_error carries its diagnostic on partial success.
    source_kind identifies the loaded result rather than an acquired Run.
    """

    tab_id: str
    data_path: str
    result_type: str
    cfg_issue: str | None
    has_analyze_params: bool
    source_kind: str = "loaded"
    analysis_error: str | None = None
    # Fixed reason for no analysis source; history is still committed successfully.
    analysis_source_issue: str | None = None


class LoadDataError(FailedPreconditionError):
    """User-facing load failure with a stable reason code."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message, reason_code=reason_code)


def _format_invalid_data_message(data_path: str, detail: str) -> str:
    return (
        "Cannot load this data file into the current tab. "
        "It may belong to a different experiment, use an older format, or have "
        "canonical axes that do not match this adapter.\n\n"
        f"File: {data_path}\n"
        f"Details: {detail}"
    )


class LoadService:
    """Synchronous canonical result load boundary.

    The service owns native file I/O, tag admission and state replacement. The Controller
    initializes analyze params afterward so run-finish and load share that policy.
    """

    def __init__(
        self, state: State, *, plan_layout: LayoutPlanner | None = None
    ) -> None:
        """Inject the pane owner; plan_layout=None leaves loaded outputs unplanned."""
        self._state = state
        self._plan_layout = plan_layout

    def load_result(self, tab_id: str, data_path: str) -> LoadTabResultOutcome:
        """Load data_path into an existing tab after GuardService.check_load.

        Call in the owner thread; data_path may be anywhere on disk. Restore
        arrays/cfg and the recorded parameter environment without live point I/O,
        location-derived identity or persistent storage. Missing/null snapshots
        provide empty content; malformed snapshots never fall back to empty.
        Metadata validates only requested fields, not unused historical sections.
        Retain native history without acquisition or Form input replacement.
        Busy or cfg notification reentrancy rejects before loading. Incompatible
        files and restore failures raise LoadDataError before replacing the pane.
        """
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        tab.cfg.require_mutation_allowed()
        logger.info("load_result: tab_id=%r data_path=%r", tab_id, data_path)
        try:
            stored = load(Path(data_path))
            metadata = metadata_content_from_container(stored)
            if metadata.experiment != tab.adapter.experiment_tag:
                raise ValueError(
                    f"Expected experiment tag {tab.adapter.experiment_tag!r}, "
                    f"got {metadata.experiment!r}"
                )
            restored = tab.adapter.restore(stored)
            metadata = restored.record.metadata
            if not metadata:
                raise ValueError("restored record has no metadata")
            snapshot = RunMetadataContent(metadata).parameter_snapshot
            parameters = (
                ParameterContext.empty()
                if snapshot is None
                else restore_parameters(snapshot)
            )
        except NotImplementedError as exc:
            raise LoadDataError(
                f"This tab does not support loading data files.\n\nDetails: {exc}",
                reason_code="unsupported_load",
            ) from exc
        except OSError as exc:
            raise LoadDataError(
                f"Could not read the data file.\n\nFile: {data_path}\nDetails: {exc}",
                reason_code="data_file_read_failed",
            ) from exc
        except ValueError as exc:
            raise LoadDataError(
                _format_invalid_data_message(data_path, str(exc)),
                reason_code="invalid_data_file",
            ) from exc

        commit_run(
            self._state,
            tab_id,
            RunPaneState(
                result=restored.record,
                source_path=data_path,
                parameters=parameters,
                storage=None,
            ),
            plan_layout=self._plan_layout,
        )
        return LoadTabResultOutcome(
            tab_id=tab_id,
            data_path=data_path,
            result_type=type(restored.record).__name__,
            cfg_issue=restored.cfg_issue,
            has_analyze_params=False,
        )
