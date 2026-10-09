from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from zcu_tools.datafile import RunSnapshot
from zcu_tools.device import DeviceInfo
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    ExpAdapterProtocol,
    RunRequest,
    require_soc_handles,
)
from zcu_tools.gui.cfg.resource import AcceptedConfig, CfgResolution, CfgRevision
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.types import ContextReadiness, ParameterContext
from zcu_tools.resources.entry import PointStorageBinding, RoleSpec

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
    observed SoC handles, device settings and fixed cfg context. accepted_cfg is
    the exact accepted publication with detached resolved values and provenance.
    RunService never re-reads these inputs from State or re-lowers a schema.
    snapshot fixes the current saved entry/point and every resolved alias at
    admission, or None for drafts or adapters without native evidence. experiment_tag is
    the adapter's persistence declaration tag, or None. Neither is recaptured
    after admission. storage retains the same saved-point persistence capability
    used for snapshot, or None for drafts, unbound tabs, missing storage or an
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
    accepted_cfg: AcceptedConfig
    snapshot: RunSnapshot | None = None
    experiment_tag: str | None = None
    storage: PointStorageBinding | None = None
    parameters: ParameterContext | None = None


class GuardService:
    """Shared domain checks and Run admission capture over State.

    View and remote controls call these checks on the owner thread before the
    corresponding service command. Successful checks return None, except
    capture_run_inputs which accepts the exact cfg revision and returns data.
    Checks reject unknown tabs or unmet domain conditions with GuardError and
    its stable reason_code; cfg revision/validity failures keep their cfg errors.
    Checks do not emit events, acquire leases or start operations.

    Analyze does not reject analysis capability NONE; post-analyze has no check
    here (see the GUI capability draft). Busy and hardware exclusion stay at
    the owning operation boundary — see ADR-0066.
    """

    def __init__(
        self,
        state: State,
        *,
        captured_parameters: Callable[
            [Mapping[str, RoleSpec], str | None], ParameterContext | None
        ],
        cfg_resolution: Callable[[ParameterContext | None], CfgResolution],
    ) -> None:
        """Use owner-sequence captures for preview-independent Run admission.

        captured_parameters freezes current point/library/storage and aliases;
        its optional target overrides only this capture. cfg_resolution must
        resolve cfg using exactly the supplied capture and published caches.
        Neither callback may poll hardware or retain live sources in its result.
        """
        self._state = state
        self._captured_parameters = captured_parameters
        self._cfg_resolution = cfg_resolution

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

    def _require_context(self, operation: str) -> None:
        """Require any editable context (DRAFT or ACTIVE), not EMPTY."""
        if self._state.session_env.readiness is ContextReadiness.EMPTY:
            raise GuardError(
                f"Cannot {operation}: no experiment context. Use Project… to set "
                "up chip/qubit or load a project.",
                reason_code="no_context",
            )

    def capture_run_inputs(
        self,
        tab_id: str,
        *,
        expected_revision: CfgRevision,
        target: str | None = None,
    ) -> RunInputs:
        """Check tab_id and capture the exact cfg revision and observed inputs.

        Requires ACTIVE context and the adapter's Run preconditions, including
        SoC when declared. Unknown tab, readiness, device settings and adapter
        validation failures raise GuardError; stale or invalid publications keep
        their cfg errors. Device settings are cached observations, not live reads.
        Captures current global parameters once for cfg context and all resolved
        alias names. target overrides this Run's first declared alias, never
        global rows or tab cfg publication. Temporary acceptance uses that same
        capture, can resolve Invalid inputs, and retains revision guards.
        Current saved-point storage supplies entry/point/description. No Run,
        lease, file or ledger write starts; source and target errors propagate.
        """
        tab = self._require_tab(tab_id)
        self._require_readiness(ContextReadiness.ACTIVE, "run")

        ctx = self._state.session_env
        parameters = self._captured_parameters(tab.adapter.parameter_roles(), target)
        # Temporary target resolution does not publish or change global aliases.
        accepted = tab.cfg.accept(
            expected_revision,
            resolution=None if target is None else self._cfg_resolution(parameters),
        )
        try:
            raw_cfg = accepted.values
            device_snapshot: dict[str, DeviceInfo] = {}
            info_type = TypeAdapter(DeviceInfo)
            for device in self._state.list_devices():
                if device.is_live():
                    if device.info is None:
                        raise ValueError(
                            f"Device {device.name!r} has no observed settings"
                        )
                    device_snapshot[device.name] = info_type.validate_python(
                        device.info
                    )
        except Exception as exc:
            raise GuardError(
                f"Config invalid: {exc}", reason_code="invalid_cfg"
            ) from exc

        req = RunRequest(
            soc=ctx.soc,
            soccfg=ctx.soccfg,
            device_snapshot=device_snapshot,
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
            tab.adapter.validate_run_request(req, raw_cfg)
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

    def check_save(self, tab_id: str) -> None:
        """Require an existing tab, ACTIVE context and a Run result.

        Raise GuardError on rejection; return None without mutation on success.
        Busy and native evidence checks remain at the Save service boundary.
        """
        tab = self._require_tab(tab_id)
        self._require_readiness(ContextReadiness.ACTIVE, "save")
        if not tab.has_run_result():
            raise GuardError(
                "No run result available to save.", reason_code="no_run_result"
            )
        logger.debug("check_save: tab_id=%r", tab_id)

    @staticmethod
    def _supports_load_data(adapter: Any) -> bool:
        """Read the import-validated load capability without probing a worker."""
        caps = getattr(adapter, "capabilities", None)
        return isinstance(caps, AdapterCapabilities) and caps.load_data

    def check_load(self, tab_id: str) -> None:
        """Require an existing tab, editable context and load_data capability.

        DRAFT and ACTIVE are editable. No SoC or result is required. Raise
        GuardError on rejection; return None without mutation on success.
        """
        tab = self._require_tab(tab_id)
        self._require_context("load data")
        if not self._supports_load_data(tab.adapter):
            raise GuardError(
                "This tab does not support loading data files.",
                reason_code="unsupported_load",
            )
        logger.debug("check_load: tab_id=%r", tab_id)

    def check_analyze(self, tab_id: str) -> None:
        """Require an existing tab, editable context and a Run result.

        Raise GuardError on rejection; return None without mutation on success.
        This check does not validate the adapter's analysis capability.
        """
        tab = self._require_tab(tab_id)
        self._require_context("analyze")
        if not tab.has_run_result():
            raise GuardError(
                "No run result available to analyze.", reason_code="no_run_result"
            )
        logger.debug("check_analyze: tab_id=%r", tab_id)

    def check_writeback(self, tab_id: str) -> None:
        """Require an existing tab, editable context and a Primary result.

        Raise GuardError on rejection; return None without mutation on success.
        This shared check does not select, apply or tear down a draft.
        """
        tab = self._require_tab(tab_id)
        self._require_context("write back")
        if not tab.has_analyze_result():
            raise GuardError(
                "No analyze result available for writeback.",
                reason_code="no_analyze_result",
            )
        logger.debug("check_writeback: tab_id=%r", tab_id)
