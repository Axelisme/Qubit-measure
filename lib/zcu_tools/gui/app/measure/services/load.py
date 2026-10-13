from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    CfgInputRejected,
    LoadDataRequest,
)
from zcu_tools.gui.cfg.resource import CfgInputError, CfgPreconditionError
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import RoleBinding

from .plot_lifecycle import release_retired_plots

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import RetiredPaneResources, State
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.parameters import ParameterContext

    from .ports import WritebackLifecyclePort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadTabResultOutcome:
    tab_id: str
    data_path: str
    result_type: str
    has_cfg_snapshot: bool
    has_analyze_params: bool
    source_kind: str = "loaded"
    cfg_backfill: Literal["applied", "not_applied"] = "not_applied"
    analysis_error: str | None = None
    # Directed Form-owned diagnostics when a native cfg cannot safely backfill.
    cfg_rejection: CfgInputRejected | None = None


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

    The service owns the adapter call and state replacement only. The Controller
    initializes analyze params afterward so run-finish and load share that policy.
    """

    def __init__(
        self,
        state: State,
        writeback: WritebackLifecyclePort,
        *,
        context: ContextControlPort,
        captured_parameters: Callable[
            [Mapping[str, str] | None], ParameterContext | None
        ],
    ) -> None:
        """Inject fixed Load-time metadata capture, distinct from the tab form capture.

        captured_parameters reads current point/library/storage and global aliases
        on the owner thread; its result must not follow later source changes.
        context reads the file's fixed point/roles in the opened entry without
        changing selection. Cfg backfill uses only the existing tab form snapshot.
        """
        self._context = context
        self._captured_parameters = captured_parameters
        self._state = state
        self._writeback = writeback

    @staticmethod
    def _supports_load_data(adapter: object) -> bool:
        """Enforce the same capability gate when a service is called directly."""
        caps = getattr(adapter, "capabilities", None)
        return isinstance(caps, AdapterCapabilities) and caps.load_data

    def load_result(self, tab_id: str, data_path: str) -> LoadTabResultOutcome:
        """Load data_path into an existing tab after GuardService.check_load.

        Call in the same owner-thread command; data_path is the selected file.
        Capture current global parameters once before the adapter call, never tab
        birth bindings. Retain its Save storage, not its aliases for analysis.
        In the same entry, read the file's point/roles once for analysis and WB.
        This reads current persisted values, not an unrecorded historical version.
        Foreign-entry/pointless/EMPTY loads have no analysis source; context-free
        analysis remains available. Historical source read failures propagate
        before result replacement. Retain native history without acquisition.
        Return committed result details and cfg backfill status. Busy or cfg
        notification reentrancy rejects before loading; adapter file/capability
        failures raise LoadDataError without replacing the previous result.
        """
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        tab.cfg.require_mutation_allowed()
        if not self._supports_load_data(tab.adapter):
            raise LoadDataError(
                "This tab does not support loading data files.",
                reason_code="unsupported_load",
            )
        request = LoadDataRequest(
            data_path=data_path,
            parameters=self._captured_parameters(None),
        )
        # Native history owns aliases; Save needs only the Load-time entry capability.
        storage = (
            None
            if request.parameters is None
            else request.parameters.bind_aliases((), {}).storage
        )
        logger.info("load_result: tab_id=%r data_path=%r", tab_id, data_path)
        try:
            result, metadata = tab.adapter.load(request)
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

        history = metadata.snapshot
        parameters = None
        if (
            request.parameters is not None
            and request.parameters.target.entry_id == str(history.entry_id)
            and history.point is not None
        ):
            parameters = self._context.read_target(
                SavedPoint(entry_id=str(history.entry_id), point=history.point),
                roles=RoleBinding(None, {}, history.roles),
            )
        snapshot = getattr(result, "cfg", None)
        retired = self._state.update_tab_loaded_result(
            tab_id,
            result,
            data_path,
            metadata=metadata,
            parameters=parameters,
            storage=storage,
        )
        self._teardown_retired(retired)
        cfg_backfill, cfg_rejection = self._backfill_cfg(tab_id, snapshot)
        return LoadTabResultOutcome(
            tab_id=tab_id,
            data_path=data_path,
            result_type=type(result).__name__,
            has_cfg_snapshot=snapshot is not None,
            has_analyze_params=False,
            cfg_backfill=cfg_backfill,
            cfg_rejection=cfg_rejection,
        )

    def _backfill_cfg(
        self, tab_id: str, snapshot: object
    ) -> tuple[Literal["applied", "not_applied"], CfgInputRejected | None]:
        if not isinstance(snapshot, ExpCfgModel):
            return "not_applied", None
        cfg = self._state.get_tab(tab_id).cfg
        try:
            rejected = cfg.load_cfg(snapshot)
            if rejected is not None:
                return "not_applied", rejected
        except (
            CfgInputError,
            CfgPreconditionError,
            ValueError,
            TypeError,
            RuntimeError,
        ):
            logger.exception("loaded config was not applied: tab_id=%r", tab_id)
            return "not_applied", None
        return "applied", None

    def _teardown_retired(self, retired: RetiredPaneResources) -> None:
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("retired load draft teardown failed")
        release_retired_plots(retired)
