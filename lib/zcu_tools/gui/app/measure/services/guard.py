from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    ExpAdapterProtocol,
    RunRequest,
    require_soc_handles,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.types import ContextReadiness, ParameterContext
from zcu_tools.resources.entry import PointStorageBinding

from .tab_form import AcceptedFormConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State


class GuardError(FailedPreconditionError):
    """Raised when a protected operation's static precondition is not met.

    This is the single failure mode both clients (View and remote) see when a
    domain check fails. It is a domain-readiness failure, never a dynamic
    resource conflict (that is OperationGate's OperationConflictError).

    ``reason_code`` is a stable machine-readable tag (e.g. ``"no_run_result"``)
    so a remote client can decide the next action without parsing the human
    ``message``. Empty when unset.
    """


@dataclass(frozen=True)
class RunInputs:
    """Run admission data captured by GuardService.capture_run_inputs.

    tab_id identifies the admitted tab. adapter_name is its registered adapter
    name; adapter is the instance that executes this Run. request holds the
    observed SoC handles and fixed cfg context. accepted_cfg is
    the exact accepted publication with detached resolved values and provenance.
    RunService never re-reads these inputs from State or re-lowers a schema.
    snapshot fixes the current saved entry/point and every resolved alias at
    admission, or None for adapters without native evidence. experiment_tag is
    the adapter's persistence declaration tag, or None. Neither is recaptured
    after admission. storage retains the same saved-point persistence capability
    used for snapshot, or None for unbound tabs, missing storage or an
    adapter without native evidence. With storage=None, Run remains usable but
    terminal does not append acquired and reports that omission in diagnostics;
    it must not claim a ledger record. With storage, terminal publishes the
    usable result before append. Append failures fail the operation with an
    ERROR diagnostic, but retain the published result for Save, without retry.
    These inputs need no release and are not a readiness proof or lease.
    The operation owns its hardware lease.
    parameters is the fixed source used at admission, or None for context-free
    work; Run publishes it with the result for later analysis.
    """

    tab_id: str
    adapter_name: str
    request: RunRequest
    adapter: ExpAdapterProtocol
    accepted_cfg: AcceptedFormConfig
    snapshot: RunSnapshot | None = None
    experiment_tag: str | None = None
    storage: PointStorageBinding | None = None
    parameters: ParameterContext | None = None


class GuardService:
    """Shared domain checks and Run admission capture over State.

    View and remote controls call these checks on the owner thread before the
    corresponding service command. Successful checks return None, except
    capture_run_inputs which accepts current Valid cfg and returns fixed data.
    Checks reject unknown tabs or unmet domain conditions with GuardError and
    its stable reason_code; cfg validity failures keep their cfg errors.
    Checks do not emit events, acquire leases or start operations.

    Analyze does not reject analysis capability NONE; post-analyze has no check
    here (see the GUI capability draft). Busy and hardware exclusion stay at
    the owning operation boundary — see ADR-0066.
    """

    def __init__(self, state: State) -> None:
        """Use State readiness and the tab resource's fixed source for Run admission.

        Construction/cfg acceptance does not capture parameters or query hardware.
        The application supplies resources before asking this owner to admit Run.
        """
        self._state = state

    def _require_tab(self, tab_id: str) -> Any:
        if not self._state.has_tab(tab_id):
            raise GuardError(f"Unknown tab: {tab_id!r}", reason_code="unknown_tab")
        return self._state.get_tab(tab_id)

    def _require_readiness(self, expected: ContextReadiness, operation: str) -> None:
        readiness = self._state.session_env.readiness
        if readiness is expected:
            return
        if expected is ContextReadiness.ACTIVE:
            raise GuardError(
                f"Cannot {operation} without an active file-backed context "
                f"(current readiness: {readiness.value}).",
                reason_code="no_active_context",
            )
        raise GuardError(
            f"Cannot {operation}: context readiness {readiness.value} "
            f"(required: {expected.value}).",
            reason_code="wrong_readiness",
        )

    def capture_run_inputs(
        self,
        tab_id: str,
    ) -> RunInputs:
        """Check tab_id and capture the current cfg and observed inputs.

        Requires ACTIVE context and the adapter's Run preconditions, including
        SoC when declared. Unknown tab, readiness and adapter validation failures
        raise GuardError; invalid publications keep their cfg errors.
        Cached device settings do not participate in admission.
        Accept current displayed values using the tab's Open/Reset capture.
        No source is captured or expression re-resolved against
        current global parameters. The fixed captured saved-point storage supplies
        entry/point/description. No Run,
        lease, file or ledger write starts; source and alias errors propagate.
        """
        tab = self._require_tab(tab_id)
        self._require_readiness(ContextReadiness.ACTIVE, "run")

        ctx = self._state.session_env
        parameters = tab.cfg.env.parameters
        accepted = tab.cfg.accept()
        req = RunRequest(
            soc=ctx.soc,
            soccfg=ctx.soccfg,
            cfg_context=None
            if parameters is None
            else parameters.library.validation_context(),
        )
        if tab.adapter.capabilities.requires_soc:
            try:
                require_soc_handles(req)
            except RuntimeError as exc:
                raise GuardError(str(exc), reason_code="no_soc") from exc

        try:
            tab.adapter.validate_run_request(req, accepted.cfg)
        except Exception as exc:
            raise GuardError(
                f"Run config invalid: {exc}", reason_code="invalid_cfg"
            ) from exc

        snapshot = None
        storage = None
        experiment_tag = tab.adapter.experiment_tag
        if (
            experiment_tag is not None
            and parameters is not None
            and parameters.storage is not None
        ):
            storage = parameters.capture_storage()
            snapshot = RunSnapshot(
                entry_id=storage.entry_id,
                entry_name=storage.entry_name,
                point=storage.point,
                description=storage.description,
                roles=storage.roles,
            )
        logger.debug("capture_run_inputs: tab_id=%r", tab_id)
        return RunInputs(
            tab_id=tab_id,
            adapter_name=tab.adapter_name,
            request=req,
            adapter=tab.adapter,
            accepted_cfg=accepted,
            snapshot=snapshot,
            experiment_tag=experiment_tag,
            storage=storage,
            parameters=parameters,
        )

    @staticmethod
    def _supports_load_data(adapter: Any) -> bool:
        """Read the import-validated load capability without probing a worker."""
        caps = getattr(adapter, "capabilities", None)
        return isinstance(caps, AdapterCapabilities) and caps.load_data

    def check_load(self, tab_id: str) -> None:
        """Require an existing tab and load_data capability.

        No live context, SoC or result is required. EMPTY loads retain native
        history without an analysis source. Raise GuardError on rejection;
        return None without mutation on success.
        """
        tab = self._require_tab(tab_id)
        if not self._supports_load_data(tab.adapter):
            raise GuardError(
                "This tab does not support loading data files.",
                reason_code="unsupported_load",
            )
        logger.debug("check_load: tab_id=%r", tab_id)

    def check_analyze(self, tab_id: str) -> None:
        """Require an existing tab and a Run result, not a live context.

        Analysis consumes the historical Run source, which may be None for
        context-free analysis. Source-dependent consumers own missing-source
        failures. Raise GuardError on rejection; return None without mutation
        on success. This check does not validate adapter analysis capability.
        """
        tab = self._require_tab(tab_id)
        if not tab.has_run_result():
            raise GuardError(
                "No run result available to analyze.", reason_code="no_run_result"
            )
        logger.debug("check_analyze: tab_id=%r", tab_id)

    def check_writeback(self, tab_id: str) -> None:
        """Require a selected working point, an existing tab and a Primary result.

        Raise GuardError on rejection; return None without mutation on success.
        This shared check does not select, apply or tear down a draft. The
        writeback owner requires a captured destination in the still-open entry
        before nonempty writes. Selection admits the command but never replaces
        its captured SavedPoint destination with the currently selected point.
        """
        tab = self._require_tab(tab_id)
        if self._state.session_env.parameters is None:
            raise GuardError(
                "Select or create a working point first", reason_code="no_context"
            )
        if not tab.has_analyze_result():
            raise GuardError(
                "No analyze result available for writeback.",
                reason_code="no_analyze_result",
            )
        logger.debug("check_writeback: tab_id=%r", tab_id)
