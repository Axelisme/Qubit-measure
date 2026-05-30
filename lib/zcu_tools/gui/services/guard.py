from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.adapter import (
    CfgSchema,
    ContextReadiness,
    ExpAdapterProtocol,
    RunRequest,
    require_soc_handles,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.state import State


class GuardError(RuntimeError):
    """Raised when a protected operation's static precondition is not met.

    This is the single failure mode both clients (View and remote) see when a
    Permit cannot be issued. It is a domain-readiness failure, never a dynamic
    resource conflict (that is OperationGate's OperationConflictError).

    ``reason_code`` is a stable machine-readable tag (e.g. ``"no_run_result"``)
    so a remote client can decide the next action without parsing the human
    ``message``. Empty when unset.
    """

    def __init__(self, message: str, *, reason_code: str = "") -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class RunPermit:
    """Proof that a run request is statically valid for ``tab_id``.

    Carries the worker payload assembled while issuing the permit (RunRequest,
    committed CfgSchema, adapter), so RunService does not re-read State. Pure
    credential — no release needed. Committed-cfg validity is verified at issue
    time by lowering once; the lowered raw is not load-bearing because the run
    worker re-lowers inside ``adapter.run(req, schema)``.
    """

    tab_id: str
    request: RunRequest
    schema: CfgSchema
    adapter: ExpAdapterProtocol


@dataclass(frozen=True)
class SavePermit:
    """Proof that a tab is eligible to save data/image (ACTIVE + has result)."""

    tab_id: str


@dataclass(frozen=True)
class AnalyzePermit:
    """Proof that a tab is eligible to analyze (context + has run result)."""

    tab_id: str


@dataclass(frozen=True)
class WritebackPermit:
    """Proof that a tab is eligible to write back (context + analyze result)."""

    tab_id: str


class GuardService:
    """Single owner of domain guard logic; issues typed Permits.

    Pure query service over ``State`` and ``ExpContext.readiness`` — no side
    effects, no event emission. Both clients must acquire a Permit before
    invoking a protected service method, so the guard logic lives in exactly
    one place and cannot drift between the View and remote paths.

    Permits cover *static* preconditions only (context readiness, committed cfg
    validity, capability requirements). *Dynamic* resource availability (tab
    busy, hardware exclusion) is checked at the operation boundary by the owning
    service via OperationGate — see CONTEXT.md and docs/adr/0001.
    """

    def __init__(self, state: "State") -> None:
        self._state = state

    def _require_tab(self, tab_id: str) -> Any:
        if tab_id not in self._state.tabs:
            raise GuardError(f"Unknown tab: {tab_id!r}", reason_code="unknown_tab")
        return self._state.get_tab(tab_id)

    def _require_readiness(self, expected: ContextReadiness, operation: str) -> None:
        readiness = self._state.exp_context.readiness
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

    def _require_context(self, operation: str) -> None:
        """Require any editable context (DRAFT or ACTIVE), not EMPTY."""
        if self._state.exp_context.readiness is ContextReadiness.EMPTY:
            raise GuardError(
                f"Cannot {operation}: no experiment context. Use Project… to set "
                "up chip/qubit or load a project.",
                reason_code="no_context",
            )

    def acquire_run_permit(self, tab_id: str) -> RunPermit:
        tab = self._require_tab(tab_id)
        self._require_readiness(ContextReadiness.ACTIVE, "run")

        ctx = self._state.exp_context
        req = RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg)

        # Lowering verifies committed cfg validity (fail-fast before any worker).
        try:
            tab.cfg_schema.to_raw_dict(req)
        except Exception as exc:
            raise GuardError(
                f"Config invalid: {exc}", reason_code="invalid_cfg"
            ) from exc

        if tab.adapter.capabilities.requires_soc:
            try:
                require_soc_handles(req)
            except RuntimeError as exc:
                raise GuardError(str(exc), reason_code="no_soc") from exc

        logger.debug("acquire_run_permit: tab_id=%r", tab_id)
        return RunPermit(
            tab_id=tab_id, request=req, schema=tab.cfg_schema, adapter=tab.adapter
        )

    def acquire_save_permit(self, tab_id: str) -> SavePermit:
        tab = self._require_tab(tab_id)
        self._require_readiness(ContextReadiness.ACTIVE, "save")
        if not tab.has_run_result():
            raise GuardError(
                "No run result available to save.", reason_code="no_run_result"
            )
        logger.debug("acquire_save_permit: tab_id=%r", tab_id)
        return SavePermit(tab_id=tab_id)

    def acquire_analyze_permit(self, tab_id: str) -> AnalyzePermit:
        tab = self._require_tab(tab_id)
        self._require_context("analyze")
        if not tab.has_run_result():
            raise GuardError(
                "No run result available to analyze.", reason_code="no_run_result"
            )
        logger.debug("acquire_analyze_permit: tab_id=%r", tab_id)
        return AnalyzePermit(tab_id=tab_id)

    def acquire_writeback_permit(self, tab_id: str) -> WritebackPermit:
        tab = self._require_tab(tab_id)
        self._require_context("write back")
        if not tab.has_analyze_result():
            raise GuardError(
                "No analyze result available for writeback.",
                reason_code="no_analyze_result",
            )
        logger.debug("acquire_writeback_permit: tab_id=%r", tab_id)
        return WritebackPermit(tab_id=tab_id)
