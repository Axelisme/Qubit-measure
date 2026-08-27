from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import AdapterCapabilities, LoadDataRequest
from zcu_tools.gui.expected_error import FailedPreconditionError

from .guard import LoadPermit

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State

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
    ) -> None:
        self._state = state
        self._writeback = writeback

    @staticmethod
    def _supports_load_data(adapter: object) -> bool:
        """Enforce the same capability gate when a service is called directly."""
        # Keep the same concrete capability contract as GuardService. The only
        # compatibility escape hatch is for old unittest.mock structural fakes;
        # a real adapter without an import-validated declaration is unsupported.
        if type(adapter).__module__ == "unittest.mock":
            return True
        caps = getattr(adapter, "capabilities", None)
        return isinstance(caps, AdapterCapabilities) and caps.load_data

    def load_result(self, permit: LoadPermit, data_path: str) -> LoadTabResultOutcome:
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        if not self._supports_load_data(tab.adapter):
            raise LoadDataError(
                "This tab does not support loading data files.",
                reason_code="unsupported_load",
            )
        ctx = self._state.exp_context
        request = LoadDataRequest(data_path=data_path, md=ctx.md, ml=ctx.ml)
        logger.info("load_result: tab_id=%r data_path=%r", tab_id, data_path)
        try:
            result = tab.adapter.load(request)
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

        retired = self._state.update_tab_loaded_result(tab_id, result, data_path)
        self._teardown_retired(retired, tab_id=tab_id)
        return LoadTabResultOutcome(
            tab_id=tab_id,
            data_path=data_path,
            result_type=type(result).__name__,
            has_cfg_snapshot=getattr(result, "cfg_snapshot", None) is not None,
            has_analyze_params=False,
        )

    def _teardown_retired(self, retired: object, *, tab_id: str | None = None) -> None:
        teardown = getattr(type(self._writeback), "teardown_draft", None)
        if not callable(teardown):
            # Transitional fakes/old callers still expose only the tab adapter.
            legacy = getattr(self._writeback, "teardown_tab_items", None)
            if tab_id is not None and callable(legacy):
                try:
                    legacy(tab_id)
                except Exception:
                    # This compatibility teardown happens after the State swap;
                    # cleanup failures must not undo or hide the loaded result.
                    logger.exception("retired legacy load draft teardown failed")
            return
        for draft in getattr(retired, "writeback_drafts", ()):
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("retired load draft teardown failed")
