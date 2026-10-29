from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.gui.app.measure.adapter import (
    AnalysisMode,
    RunRequest,
    require_soc_handles,
)
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.registry import RegisteredAdapter
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry import StorageBinding

from .tab_form import AcceptedFormConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State


class TabAction(str, Enum):
    """The tab operations whose domain readiness is owned by GuardService."""

    RUN = "run"
    LOAD = "load"
    ANALYZE = "analyze"
    POST_ANALYZE = "post_analyze"
    WRITEBACK = "writeback"


@dataclass(frozen=True)
class AdmissionDecision:
    """One owner-thread observation of tab readiness, not a hardware lease.

    allowed is True only when all action prerequisites pass. reason is None
    on success, otherwise the first stable wire reason code from admit's
    documented order. message is empty on success; on rejection it is the
    shared tooltip/error text and must not be parsed for the reason code.
    """

    allowed: bool
    reason: str | None
    message: str


class GuardError(FailedPreconditionError):
    """Raised when a protected operation's admission precondition is not met.

    This is the single failure mode both clients (View and remote) see when a
    domain check fails. Tab busy is a domain-readiness failure; hardware
    conflicts remain OperationGate's OperationConflictError.

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
    adapter: RegisteredAdapter
    accepted_cfg: AcceptedFormConfig
    snapshot: RunSnapshot | None = None
    experiment_tag: str | None = None
    storage: StorageBinding | None = None
    parameters: ParameterContext | None = None


class GuardService:
    """Observe domain readiness and capture admitted Run inputs over State.

    Call on the owner thread. admit returns a decision without mutation;
    check_* and capture_run_inputs raise GuardError for that decision's rejection.
    Capture accepts cfg only after admission. No method emits events, starts
    operations or acquires hardware leases. Tab busy belongs here; startup and
    competing hardware operations remain with the gate (ADR-0066).
    """

    def __init__(self, state: State, *, access: ExperimentAccess) -> None:
        """Use State readiness and the tab resource's fixed source for Run admission.

        access is the application's shared catalog reload/shutdown gate. The
        application supplies resources before asking this owner to admit Run.
        Construction and admission do not capture sources or query hardware.
        """
        self._state = state
        self._access = access

    def admit(self, tab_id: str, action: TabAction) -> AdmissionDecision:
        """Read the first domain blocker without starting work or acquiring leases.

        All actions first check experiment_unavailable, then unknown_tab.
        Remaining reason codes, in first-failure order, are:

        * RUN: no_active_context (wrong_readiness for other required readiness),
          not_valid (cfg preparation, model, then lowering), no_soc, tab_busy.
        * LOAD: tab_busy. No context, SoC, result or storage is required.
        * ANALYZE: no_run_result, no_analysis, tab_busy.
        * POST_ANALYZE: no_analyze_result, no_post_analysis, tab_busy.
        * WRITEBACK: no_context, no_analyze_result (Primary), tab_busy.

        Call on the owner thread. Success has reason=None and message="".
        Rejection messages are the shared tooltip and GuardError text; reason
        codes are the wire tags. No Run/Load storage prerequisite is added;
        Writeback validates its captured destination in its owner. Cfg reads
        may raise their resource-lifetime or author lower-hook exceptions.
        Hardware exclusion remains with the operation gate. Unknown actions
        raise ValueError after the availability and tab checks.
        """
        try:
            self._access.require_available()
        except FailedPreconditionError as exc:
            return AdmissionDecision(False, "experiment_unavailable", str(exc))
        if not self._state.has_tab(tab_id):
            return AdmissionDecision(False, "unknown_tab", f"Unknown tab: {tab_id!r}")
        match action:
            case TabAction.RUN:
                blocker = self._run_blocker(tab_id)
            case TabAction.ANALYZE | TabAction.POST_ANALYZE:
                blocker = self._analysis_blocker(tab_id, action)
            case TabAction.WRITEBACK:
                blocker = self._writeback_blocker(tab_id)
            case TabAction.LOAD:
                blocker = None
            case _:
                raise ValueError(f"Unknown tab action: {action!r}")
        if blocker is not None:
            return blocker
        if self._state.is_tab_busy(tab_id):
            return AdmissionDecision(False, "tab_busy", f"Tab {tab_id!r} is busy")
        return AdmissionDecision(True, None, "")

    def _analysis_blocker(
        self, tab_id: str, action: TabAction
    ) -> AdmissionDecision | None:
        tab = self._state.get_tab(tab_id)
        if action is TabAction.ANALYZE:
            if not tab.has_run_result():
                return AdmissionDecision(
                    False, "no_run_result", "No run result available to analyze."
                )
            if tab.adapter.capabilities.analysis is AnalysisMode.NONE:
                return AdmissionDecision(
                    False, "no_analysis", "This experiment has no analysis."
                )
        else:
            if not tab.has_analyze_result():
                return AdmissionDecision(
                    False,
                    "no_analyze_result",
                    f"Tab {tab_id!r} has no primary analyze result to post-analyze",
                )
            if not tab.adapter.capabilities.post_analysis:
                return AdmissionDecision(
                    False, "no_post_analysis", "This experiment has no post-analysis."
                )
        return None

    def _writeback_blocker(self, tab_id: str) -> AdmissionDecision | None:
        if self._state.session_env.parameters is None:
            return AdmissionDecision(
                False, "no_context", "Select or create a working point first"
            )
        if not self._state.get_tab(tab_id).has_analyze_result():
            return AdmissionDecision(
                False, "no_analyze_result", "No analyze result available for writeback."
            )
        return None

    def _run_blocker(self, tab_id: str) -> AdmissionDecision | None:
        try:
            self._require_active_context("run")
        except GuardError as exc:
            return AdmissionDecision(False, exc.reason_code, str(exc))
        tab = self._state.get_tab(tab_id)
        reason = tab.cfg.run_invalid_reason()
        if reason is not None:
            return AdmissionDecision(False, "not_valid", reason)
        ctx = self._state.session_env
        try:
            require_soc_handles(
                RunRequest(soc=ctx.soc, soccfg=ctx.soccfg, cfg_context=None)
            )
        except RuntimeError as exc:
            return AdmissionDecision(False, "no_soc", str(exc))
        return None

    def require_admission(self, tab_id: str, action: TabAction) -> None:
        """Require tab_id's domain readiness for the requested TabAction.

        Call admit(tab_id, action) exactly once on the owner thread, without
        mutation. Return None when allowed; otherwise raise GuardError with
        the decision's message and reason_code (an empty string if reason is
        None). Other exceptions from admit propagate unchanged.
        This read-only observation does not capture inputs, start work or
        acquire a lease. Success does not guarantee later operation admission
        or hardware availability; the operation gate still owns exclusion.
        """
        decision = self.admit(tab_id, action)
        if not decision.allowed:
            raise GuardError(decision.message, reason_code=decision.reason or "")

    def _require_active_context(self, operation: str) -> None:
        context = self._state.session_env
        if not context.is_active():
            raise GuardError(
                f"Cannot {operation} without an active file-backed context "
                f"(current readiness: {context.readiness.value}).",
                reason_code="no_active_context",
            )

    def capture_run_inputs(
        self,
        tab_id: str,
    ) -> RunInputs:
        """Check tab_id and capture the current cfg and observed inputs.

        First read admit(RUN) once; every rejection raises GuardError with
        exactly that decision's reason and message, including invalid cfg.
        After admission, accept cfg once. Execution-time lifetime/reentrancy
        failures from accept retain their CfgPreconditionError.
        Cached device settings do not participate in admission.
        Accept current displayed values using the tab's Open/Reset capture.
        No source is captured or expression re-resolved against
        current global parameters. The fixed captured saved-point storage supplies
        entry/point/description. No Run,
        lease, file or ledger write starts; source and alias errors propagate.
        """
        self.require_admission(tab_id, TabAction.RUN)
        tab = self._state.get_tab(tab_id)

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
        snapshot = None
        storage = None
        experiment_tag = tab.adapter.experiment_tag
        if parameters is not None and parameters.storage is not None:
            storage = parameters.capture_storage()
            snapshot = RunSnapshot(
                entry_name=storage.entry_name, point=storage.point, roles=storage.roles
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

    def check_load(self, tab_id: str) -> None:
        """Permit every registered adapter to attempt Load on an existing tab.

        No live context, SoC or result is required. EMPTY loads retain native
        history without an analysis source. Raise GuardError on rejection;
        return None without mutation on success.
        """
        self.require_admission(tab_id, TabAction.LOAD)

    def check_analyze(self, tab_id: str) -> None:
        """Require an existing tab and a Run result, not a live context.

        Analysis consumes the historical Run source, which may be None for
        context-free analysis. Source-dependent consumers own missing-source
        failures. Raise GuardError on rejection; return None without mutation
        on success. AnalysisMode.NONE and tab busy are rejected.
        """
        self.require_admission(tab_id, TabAction.ANALYZE)

    def check_writeback(self, tab_id: str) -> None:
        """Require a selected working point, an existing tab and a Primary result.

        Raise GuardError on rejection; return None without mutation on success.
        This shared check does not select, apply or tear down a draft. The
        writeback owner requires a captured destination in the still-open entry
        before nonempty writes. Selection admits the command but never replaces
        its captured SavedPoint destination with the currently selected point.
        """
        self.require_admission(tab_id, TabAction.WRITEBACK)
