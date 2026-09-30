from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, LoadDataRequest
from zcu_tools.gui.app.measure.adapter.loaded_cfg import project_loaded_cfg
from zcu_tools.gui.cfg.resource import CfgInputError, CfgPreconditionError
from zcu_tools.gui.expected_error import FailedPreconditionError

from .guard import LoadPermit

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import RetiredPaneResources, State

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
        provide_options: Callable[[str], Sequence[object]],
    ) -> None:
        self._state = state
        self._writeback = writeback
        self._provide_options = provide_options

    @staticmethod
    def _supports_load_data(adapter: object) -> bool:
        """Enforce the same capability gate when a service is called directly."""
        caps = getattr(adapter, "capabilities", None)
        return isinstance(caps, AdapterCapabilities) and caps.load_data

    def load_result(self, permit: LoadPermit, data_path: str) -> LoadTabResultOutcome:
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        tab.cfg.require_mutation_allowed()
        if not self._supports_load_data(tab.adapter):
            raise LoadDataError(
                "This tab does not support loading data files.",
                reason_code="unsupported_load",
            )
        ctx = self._state.session_env
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
        self._teardown_retired(retired)
        return LoadTabResultOutcome(
            tab_id=tab_id,
            data_path=data_path,
            result_type=type(result).__name__,
            has_cfg_snapshot=getattr(result, "cfg_snapshot", None) is not None,
            has_analyze_params=False,
            cfg_backfill=self._backfill_cfg(
                tab_id, getattr(result, "cfg_snapshot", None)
            ),
        )

    def _backfill_cfg(
        self, tab_id: str, snapshot: object
    ) -> Literal["applied", "not_applied"]:
        if not isinstance(snapshot, ExpCfgModel):
            return "not_applied"
        cfg = self._state.get_tab(tab_id).cfg
        revision = cfg.observe().ref.revision
        try:
            candidate = project_loaded_cfg(
                cfg.snapshot_inputs(), snapshot, provide_options=self._provide_options
            )
            if candidate is None:
                return "not_applied"
            cfg.replace_inputs(revision, candidate, require_valid=True)
        except (
            CfgInputError,
            CfgPreconditionError,
            ValueError,
            TypeError,
            RuntimeError,
        ):
            logger.exception("loaded config was not applied: tab_id=%r", tab_id)
            return "not_applied"
        return "applied"

    def _teardown_retired(self, retired: RetiredPaneResources) -> None:
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("retired load draft teardown failed")
