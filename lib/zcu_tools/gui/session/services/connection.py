from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.ports import ExclusionGate, OperationKind
from zcu_tools.gui.session.types import SocCfgHandle, SocHandle

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner
    from zcu_tools.gui.session.state import SessionState


# ---------------------------------------------------------------------------
# Typed requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectMockRequest:
    """Connect to a local in-process MockSoc.

    ``sim_params`` is an optional ``SimParams`` override (typed ``Any`` to avoid an
    import-time dependency on the sim package). The production GUI never sets it, so
    the mock keeps yielding ``DEFAULT_SIMPARAM`` data; tests pass a high-snr params
    to cut the reps/rounds a fittable decay needs (the snr only changes per-shot
    noise, so the sim-predictor provisioning stays consistent).
    """

    sim_params: Any = None


@dataclass(frozen=True)
class ConnectRemoteRequest:
    """Connect to a remote ZCU board via SocProxy."""

    ip: str
    port: int


ConnectRequest = ConnectMockRequest | ConnectRemoteRequest


# ---------------------------------------------------------------------------
# Typed expected failures
# ---------------------------------------------------------------------------


class SoCConnectionError(RuntimeError):
    """Expected failure: SoC connection attempt rejected by user environment."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SoCConnectionService(QObject):
    """Owns SoC connect lifecycle, delegating async work to OperationRunner.

    Connect work is always reported via Qt signals (connection_finished /
    connection_failed) so the View has a single non-blocking contract. Both mock
    and remote connects run via bg.submit, so signals arrive on the next
    event-loop tick after the bg work finishes — preserving the async contract.
    """

    connection_finished: Signal = Signal()
    connection_failed: Signal = Signal(str)

    def __init__(
        self,
        state: SessionState,
        bus: BaseEventBus,
        gate: ExclusionGate,
        handles: OperationHandles,
        runner: OperationRunner,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._bus = bus
        # Connect composes both leaves (ADR-0019): an Exclusion lease
        # (SOC_CONNECT vs run / another connect) + a Handle. A connect has no
        # cancellation point, so cancel_hook=None (§B.3 equivalence).
        self._gate = gate
        self._handles = handles
        self._runner = runner
        # _active_token: POST-BEGIN set, on_terminal cleared (run.py pattern).
        self._active_token: int | None = None
        self._pending_is_mock: bool = False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_soc(self) -> bool:
        return self._state.exp_context.soc is not None

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._state.exp_context.soccfg

    def is_mock_soc(self) -> bool:
        """Whether the current connection is the offline mock board.

        Reflects the last connect request's kind; only meaningful while a SoC is
        connected (a successful connect keeps soc set, so pair with has_soc).
        """
        return self._pending_is_mock

    def is_connect_active(self) -> bool:
        # _active_token is POST-BEGIN set and cleared in on_terminal (§B.4 option A,
        # mirrors run.py _active_token pattern).
        return self._active_token is not None

    # ------------------------------------------------------------------
    # Connect (async-shaped API, OperationRunner client)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_connect_work(req: ConnectRequest) -> tuple[SocHandle, SocCfgHandle]:
        """Do the actual SoC connect, returning the (soc, soccfg) handles.

        The single source of truth for "how a connect is performed", shared by the
        async ``start_connect`` (off-main via the runner) and the synchronous
        ``connect_sync`` (on-main via the wire). It does only the connect itself —
        no State writes, no version bump, no event emit (those are the caller's job
        via ``_apply_connection``), so neither caller's threading model leaks in.
        """
        if isinstance(req, ConnectMockRequest):
            from zcu_tools.program.v2.mocksoc import make_mock_soc
            from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM

            sim = req.sim_params if req.sim_params is not None else DEFAULT_SIMPARAM
            return make_mock_soc(sim=sim)
        # remote
        try:
            from zcu_tools.remote import make_soc_proxy
        except ImportError as exc:
            raise SoCConnectionError(
                f"Cannot import ZCU client libraries: {exc}. "
                "Use MockSoc for offline testing."
            ) from exc
        return make_soc_proxy(req.ip, req.port)

    def connect_sync(self, req: ConnectRequest) -> tuple[SocHandle, SocCfgHandle]:
        """Connect synchronously on the calling (Qt main) thread.

        The wire path (``soc.connect``) wants a single blocking call that returns
        the connected SoC directly, with all post-connect side effects applied
        before it returns — so it runs on-main and the IO worker blocks on it. It
        still holds the same SOC_CONNECT exclusion lease as the async path (so the
        wire connect and the GUI's connect button cannot race), minting a token
        purely as the lease handle (no async await: the work runs inline here, not
        on a bg thread). On success it calls the shared ``_apply_connection`` — the
        identical side-effect routine the async finished-handler uses — so the
        FLUX-AWARE-MOCK provisioning (driven off the emitted SocChangedPayload),
        the State write and the soc version bump are byte-for-byte the same.
        """
        if not isinstance(req, (ConnectMockRequest, ConnectRemoteRequest)):
            raise TypeError(f"Unsupported connect request: {type(req).__name__}")

        is_mock = isinstance(req, ConnectMockRequest)
        # Hold the SOC_CONNECT lease for the whole synchronous connect so it cannot
        # race the GUI's connect button (or an async start_connect). A connect has
        # no cancellation point, so the token is a lease handle only (cancel_hook
        # None); it is settled + released in the finally regardless of outcome.
        self._gate.ensure_can_start(OperationKind.SOC_CONNECT, resource_id=None)
        token = self._handles.create(cancel_hook=None, origin=self._bus.current_origin)
        self._gate.register(token, OperationKind.SOC_CONNECT, owner_id="soc")
        self._active_token = token
        self._pending_is_mock = is_mock
        with self._bus.origin(self._handles.event_origin(token)):
            try:
                soc, soccfg = self._run_connect_work(req)
                self._apply_connection(soc, soccfg, is_mock)
                self._handles.settle(token, OperationOutcome("finished"))
                return soc, soccfg
            except Exception as exc:
                self._handles.settle(
                    token, OperationOutcome("failed", _format_error(exc))
                )
                raise
            finally:
                self._active_token = None
                self._gate.release(token)

    def start_connect(self, req: ConnectRequest) -> int:
        """Start a SoC connect; outcome always arrives via signals.

        Delegates to OperationRunner with cancel_hook=None (connect has no
        cancellation point — §B.3 equivalence with old _ConnectWorker). Returns
        the operation token minted by runner.begin.
        """
        if not isinstance(req, (ConnectMockRequest, ConnectRemoteRequest)):
            raise TypeError(f"Unsupported connect request: {type(req).__name__}")

        is_mock = isinstance(req, ConnectMockRequest)

        def work(factory: Any) -> tuple[SocHandle, SocCfgHandle]:
            # factory is None (wants_progress=False). Both branches run off-main via
            # bg.submit, preserving the async signal contract for mock and remote.
            return self._run_connect_work(req)

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            # Runs on main thread (bg marshal). _apply_connection writes State and
            # calls version.bump("soc") — main-thread invariant preserved (§B.2).
            self._active_token = None
            if bg.ok:
                soc, soccfg = bg.result
                self._apply_connection(soc, soccfg, is_mock)
                settle(OperationOutcome("finished"))
                self.connection_finished.emit()
            else:
                assert bg.error is not None
                msg = _format_error(bg.error)
                settle(OperationOutcome("failed", msg))
                self.connection_failed.emit(msg)

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.SOC_CONNECT,
                owner_id="soc",
            ),
            owner_id="soc",
            wants_progress=False,
            cancel_hook=None,
            work=work,
            run_in_pool=False,  # dedicated thread, matching old _ConnectWorker
            on_terminal=on_terminal,
        )

        token = self._runner.begin(spec)
        # POST-BEGIN: set bookkeeping only after begin() succeeds (never written
        # on conflict, mirrors run.py _active_token pattern).
        self._active_token = token
        self._pending_is_mock = is_mock
        return token

    def _apply_connection(
        self, soc: SocHandle, soccfg: SocCfgHandle, is_mock: bool
    ) -> None:
        logger.info("connect succeeded: mock=%s", is_mock)
        new_ctx = dataclasses.replace(self._state.exp_context, soc=soc, soccfg=soccfg)
        self._state.set_context(new_ctx)
        # soc is its own resource (a run depends on it independently of context);
        # bump it here, on the main thread, where the soc is written. Deliberately
        # NOT bumping ``context``: set_context no longer does so, and a soc connect
        # does not change md/ml — bumping context would spuriously mark md/ml-
        # dependent ops (run / editor.commit / writeback) stale.
        self._state.version.bump("soc")
        self._bus.emit(
            SocChangedPayload(soc=soc, soccfg=soccfg, is_mock=is_mock),
        )


def _format_error(exc: Exception) -> str:
    """Classify connection exceptions to user-facing prefixes (§B.5).

    Mirrors the old _ConnectWorker.run error classification so MCP / View error
    messages stay stable after the runner migration.
    """
    if isinstance(exc, (ConnectionRefusedError, TimeoutError, OSError)):
        return f"Connection failed: {exc}"
    return f"Connection error: {exc}"
